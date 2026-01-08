# --- llm_processor.py ---
import json
import queue
import re
import threading
import time
from typing import Any, ClassVar

from loguru import logger
from pydantic import HttpUrl  # If HttpUrl is used by config
import requests

# Import tools
from ..tools import VolumeControlTool, WebSearchTool


class LanguageModelProcessor:
    """
    A thread that processes text input for a language model, streaming responses and sending them to TTS.
    This class is designed to run in a separate thread, continuously checking for new text to process
    until a shutdown event is set. It handles conversation history, manages streaming responses,
    and sends synthesized sentences to a TTS queue.
    """

    PUNCTUATION_SET: ClassVar[set[str]] = {".", "!", "?", ":", ";", "?!", "\n", "\n\n"}

    def __init__(
        self,
        llm_input_queue: queue.Queue[str],
        tts_input_queue: queue.Queue[str],
        conversation_history: list[dict[str, str]],  # Shared
        completion_url: HttpUrl,
        model_name: str,  # Renamed from 'model' to avoid conflict
        api_key: str | None,
        processing_active_event: threading.Event,  # To check if we should stop streaming
        shutdown_event: threading.Event,
        pause_time: float = 0.05,
    ) -> None:
        self.llm_input_queue = llm_input_queue
        self.tts_input_queue = tts_input_queue
        self.conversation_history = conversation_history
        self.completion_url = completion_url
        self.model_name = model_name
        self.api_key = api_key
        self.processing_active_event = processing_active_event
        self.shutdown_event = shutdown_event
        self.pause_time = pause_time

        self.prompt_headers = {"Content-Type": "application/json"}
        if api_key:
            self.prompt_headers["Authorization"] = f"Bearer {api_key}"

        # Initialize tools
        self.volume_tool = VolumeControlTool()
        self.web_search_tool = WebSearchTool()
        self.tools = {
            "volume": self.volume_tool,
            "web_search": self.web_search_tool
        }

    def _detect_tool_call(self, text: str) -> tuple[str, Any] | None:
        """
        Detect if the input text contains a tool call.
        
        Args:
            text: Input text from user
            
        Returns:
            Tuple of (tool_name, tool_instance) if tool detected, None otherwise
        """
        text_lower = text.lower()
        
        # Volume control patterns
        volume_patterns = [
            r"volume\s+(up|down|increase|decrease|raise|lower|set|adjust)",
            r"(up|down|increase|decrease|raise|lower)\s+volume",
            r"set\s+volume\s+to",
            r"adjust\s+volume",
            r"make\s+(it|volume)\s+(louder|quieter|softer|higher|lower)",
            r"turn\s+(it|volume)\s+(up|down)",
        ]
        
        # Web search patterns
        search_patterns = [
            r"search\s+the\s+web\s+for",
            r"search\s+web\s+for",
            r"search\s+for",
            r"web\s+search",
            r"google\s+search",
        ]
        
        # Check volume patterns first (more specific)
        for pattern in volume_patterns:
            if re.search(pattern, text_lower):
                return "volume", self.volume_tool
        
        # Check search patterns
        for pattern in search_patterns:
            if re.search(pattern, text_lower):
                return "web_search", self.web_search_tool
        
        return None

    def _execute_tool(self, tool_name: str, tool: Any, command: str) -> dict[str, Any]:
        """
        Execute a tool command.
        
        Args:
            tool_name: Name of the tool
            tool: Tool instance
            command: Original user command
            
        Returns:
            Tool execution result
        """
        try:
            logger.info(f"Executing tool '{tool_name}' with command: '{command}'")
            result = tool.execute(command)
            
            if result.get("success"):
                # Generate GLaDOS-style response for successful tool execution
                if tool_name == "volume":
                    action = "adjusted"
                    if "set" in command.lower():
                        action = "set"
                    elif "up" in command.lower() or "increase" in command.lower():
                        action = "increased"
                    elif "down" in command.lower() or "decrease" in command.lower():
                        action = "decreased"
                    
                    response = f"Volume {action} to {result.get('volume', 'unknown')}%. Another trivial task completed."
                elif tool_name == "web_search":
                    # For web search, use the actual search response
                    response = result.get('response', "Search completed but no results found.")
                else:
                    response = f"Tool '{tool_name}' executed successfully."
                
                logger.success(f"Tool execution successful: {response}")
                return {"success": True, "response": response, "tool_result": result}
            else:
                # Error response
                error_msg = result.get("error", "Unknown error")
                response = f"Tool execution failed: {error_msg}. How disappointing."
                logger.error(f"Tool execution failed: {error_msg}")
                return {"success": False, "response": response, "tool_result": result}
                
        except Exception as e:
            error_msg = f"Unexpected error executing tool: {e}"
            logger.exception(error_msg)
            response = "The tool malfunctioned. How utterly predictable."
            return {"success": False, "response": response, "tool_result": {"error": str(e)}}

    def _clean_raw_bytes(self, line: bytes) -> dict[str, str] | None:
        """
        Clean and parse a raw byte line from the LLM response.
        Handles both OpenAI and Ollama formats, returning a dictionary or None if parsing fails.

        Args:
            line (bytes): The raw byte line from the LLM response.
        Returns:
            dict[str, str] | None: Parsed JSON dictionary or None if parsing fails.
        """
        try:
            # Handle OpenAI format
            if line.startswith(b"data: "):
                json_str = line.decode("utf-8")[6:]
                if json_str.strip() == "[DONE]":  # Handle OpenAI [DONE] marker
                    return {"done_marker": "True"}
                parsed_json: dict[str, Any] = json.loads(json_str)
                return parsed_json
            # Handle Ollama format
            else:
                parsed_json = json.loads(line.decode("utf-8"))
                if isinstance(parsed_json, dict):
                    return parsed_json
                return None
        except json.JSONDecodeError:
            # If it's not JSON, it might be Ollama's final summary object which isn't part of the stream
            # Or just noise.
            logger.trace(
                f"LLM Processor: Failed to parse non-JSON server response line: "
                f"{line[:100].decode('utf-8', errors='replace')}"
            )  # Log only a part
            return None
        except Exception as e:
            logger.warning(
                f"LLM Processor: Failed to parse server response: {e} for line: "
                f"{line[:100].decode('utf-8', errors='replace')}"
            )
            return None

    def _process_chunk(self, line: dict[str, Any]) -> str | None:
        # Copy from Glados._process_chunk
        if not line or not isinstance(line, dict):
            return None
        try:
            # Handle OpenAI format
            if line.get("done_marker"):  # Handle [DONE] marker
                return None
            elif "choices" in line:  # OpenAI format
                content = line.get("choices", [{}])[0].get("delta", {}).get("content")
                return str(content) if content else None
            # Handle Ollama format
            else:
                content = line.get("message", {}).get("content")
                return content if content else None
        except Exception as e:
            logger.error(f"LLM Processor: Error processing chunk: {e}, chunk: {line}")
            return None

    def _process_sentence_for_tts(self, current_sentence_parts: list[str]) -> None:
        """
        Process the current sentence parts and send the complete sentence to the TTS queue.
        Cleans up the sentence by removing unwanted characters and formatting it for TTS.
        Args:
            current_sentence_parts (list[str]): List of sentence parts to be processed.
        """
        sentence = "".join(current_sentence_parts)
        sentence = re.sub(r"\*.*?\*|\(.*?\)", "", sentence)
        sentence = sentence.replace("\n\n", ". ").replace("\n", ". ").replace("  ", " ").replace(":", " ")

        if sentence and sentence != ".":  # Avoid sending just a period
            logger.info(f"LLM Processor: Sending to TTS queue: '{sentence}'")
            self.tts_input_queue.put(sentence)

    def run(self) -> None:
        """
        Starts the main loop for the LanguageModelProcessor thread.

        This method continuously checks the LLM input queue for text to process.
        It processes the text, sends it to the LLM API, and streams the response.
        It handles conversation history, manages streaming responses, and sends synthesized sentences
        to a TTS queue. The thread will run until the shutdown event is set, at which point it will exit gracefully.
        """
        logger.info("LanguageModelProcessor thread started.")
        while not self.shutdown_event.is_set():
            try:
                detected_text = self.llm_input_queue.get(timeout=self.pause_time)
                if not self.processing_active_event.is_set():  # Check if we were interrupted before starting
                    logger.info("LLM Processor: Interruption signal active, discarding LLM request.")
                    # Ensure EOS is sent if a previous stream was cut short by this interruption
                    # This logic might need refinement based on state. For now, assume no prior stream.
                    continue

                logger.info(f"LLM Processor: Received text for LLM: '{detected_text}'")
                
                # Check for tool calls before sending to LLM
                tool_call = self._detect_tool_call(detected_text)
                if tool_call:
                    tool_name, tool_instance = tool_call
                    logger.info(f"Tool call detected: {tool_name}")
                    
                    # Execute tool and send response directly to TTS
                    tool_result = self._execute_tool(tool_name, tool_instance, detected_text)
                    response_text = tool_result["response"]
                    
                    # Add to conversation history
                    self.conversation_history.append({"role": "user", "content": detected_text})
                    self.conversation_history.append({"role": "assistant", "content": response_text})
                    
                    # Send response to TTS
                    self.tts_input_queue.put(response_text)
                    self.tts_input_queue.put("<EOS>")
                    continue
                
                # No tool detected, proceed with normal LLM processing
                self.conversation_history.append({"role": "user", "content": detected_text})

                data = {
                    "model": self.model_name,
                    "stream": True,
                    "messages": self.conversation_history,
                    # Add other parameters like temperature, max_tokens if needed from config
                }

                sentence_buffer: list[str] = []
                try:
                    with requests.post(
                        str(self.completion_url),
                        headers=self.prompt_headers,
                        json=data,
                        stream=True,
                        timeout=30,  # Add a timeout for the request itself
                    ) as response:
                        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                        logger.debug("LLM Processor: Request to LLM successful, processing stream...")
                        for line in response.iter_lines():
                            if not self.processing_active_event.is_set() or self.shutdown_event.is_set():
                                logger.info("LLM Processor: Interruption or shutdown detected during LLM stream.")
                                break  # Stop processing stream

                            if line:
                                cleaned_line_data = self._clean_raw_bytes(line)
                                if cleaned_line_data:
                                    chunk = self._process_chunk(cleaned_line_data)
                                    if chunk:  # Chunk can be an empty string, but None means no actual content
                                        sentence_buffer.append(chunk)
                                        # Split on defined punctuation or if chunk itself is punctuation
                                        if chunk.strip() in self.PUNCTUATION_SET and (
                                            len(sentence_buffer) < 2 or not sentence_buffer[-2].strip().isdigit()
                                        ):
                                            self._process_sentence_for_tts(sentence_buffer)
                                            sentence_buffer = []
                                    # OpenAI [DONE]
                                    elif cleaned_line_data.get("done_marker"):  # OpenAI [DONE]
                                        break
                                    # Ollama end
                                    elif cleaned_line_data.get("done") and cleaned_line_data.get("response") == "":
                                        break

                        # After loop, process any remaining buffer content if not interrupted
                        if self.processing_active_event.is_set() and sentence_buffer:
                            self._process_sentence_for_tts(sentence_buffer)

                except requests.exceptions.ConnectionError as e:
                    logger.error(f"LLM Processor: Connection error to LLM service: {e}")
                    self.tts_input_queue.put(
                        "I'm unable to connect to my thinking module. Please check the LLM service connection."
                    )
                except requests.exceptions.Timeout as e:
                    logger.error(f"LLM Processor: Request to LLM timed out: {e}")
                    self.tts_input_queue.put("My brain seems to be taking too long to respond. It might be overloaded.")
                except requests.exceptions.HTTPError as e:
                    status_code = (
                        e.response.status_code
                        if hasattr(e, "response") and hasattr(e.response, "status_code")
                        else "unknown"
                    )
                    logger.error(f"LLM Processor: HTTP error {status_code} from LLM service: {e}")
                    self.tts_input_queue.put(f"I received an error from my thinking module. HTTP status {status_code}.")
                except requests.exceptions.RequestException as e:
                    logger.error(f"LLM Processor: Request to LLM failed: {e}")
                    self.tts_input_queue.put("Sorry, I encountered an error trying to reach my brain.")
                except Exception as e:
                    logger.exception(f"LLM Processor: Unexpected error during LLM request/streaming: {e}")
                    self.tts_input_queue.put("I'm having a little trouble thinking right now.")
                finally:
                    # Always send EOS if we started processing, unless interrupted early
                    if self.processing_active_event.is_set():  # Only send EOS if not interrupted
                        logger.debug("LLM Processor: Sending EOS token to TTS queue.")
                        self.tts_input_queue.put("<EOS>")
                    else:
                        logger.info("LLM Processor: Interrupted, not sending EOS from LLM processing.")
                        # The AudioPlayer will handle clearing its state.
                        # If an EOS was already sent by TTS from a *previous* partial sentence,
                        # this could lead to an early clear of currently_speaking.
                        # The `processing_active_event` is key to synchronize.

            except queue.Empty:
                pass  # Normal
            except Exception as e:
                logger.exception(f"LLM Processor: Unexpected error in main run loop: {e}")
                time.sleep(0.1)
        logger.info("LanguageModelProcessor thread finished.")
