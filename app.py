import uuid
from datetime import datetime
import json
from pathlib import Path
import time

import requests
import streamlit as st

# Set page configuration
st.set_page_config(page_title="My AI Chat", layout="wide")

# App title
st.title("My AI Chat")

# Load Hugging Face token from secrets
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    hf_token = ""

# Check if token is missing or empty
if not hf_token:
    st.error("Hugging Face token is missing. Please add it to .streamlit/secrets.toml as HF_TOKEN.")
    st.stop()

# API endpoint and model
endpoint = "https://router.huggingface.co/v1/chat/completions"
model = "meta-llama/Llama-3.2-1B-Instruct"
chats_folder = Path("chats")
memory_file = Path("memory.json")
request_timeout = 60


# Create a new chat with basic starter information
def create_chat():
    timestamp = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    return {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "timestamp": timestamp,
        "messages": [],
    }


# Create the chats folder if it does not exist
def ensure_chats_folder():
    chats_folder.mkdir(exist_ok=True)


# Build a file path for one chat
def get_chat_file_path(chat_id):
    return chats_folder / f"{chat_id}.json"


# Load saved user memory from disk
def load_memory_from_disk():
    if not memory_file.exists():
        return {}

    try:
        with memory_file.open("r", encoding="utf-8") as file:
            memory_data = json.load(file)

        if isinstance(memory_data, dict):
            return memory_data
    except (json.JSONDecodeError, OSError):
        st.warning("Could not read memory.json. Starting with empty memory.")

    return {}


# Save user memory to disk
def save_memory_to_disk(memory_data):
    with memory_file.open("w", encoding="utf-8") as file:
        json.dump(memory_data, file, indent=2)


# Set up user memory in session state
def initialize_user_memory():
    if "user_memory" not in st.session_state:
        st.session_state.user_memory = load_memory_from_disk()


# Load all saved chats from disk
def load_chats_from_disk():
    ensure_chats_folder()
    loaded_chats = []

    for chat_file in sorted(chats_folder.glob("*.json")):
        try:
            with chat_file.open("r", encoding="utf-8") as file:
                chat_data = json.load(file)

            if "id" in chat_data and "messages" in chat_data:
                chat_data.setdefault("title", "New Chat")
                chat_data.setdefault("timestamp", "Unknown Time")
                loaded_chats.append(chat_data)
        except (json.JSONDecodeError, OSError):
            st.warning(f"Could not load chat file: {chat_file.name}")

    return loaded_chats


# Save one chat to its own JSON file
def save_chat_to_disk(chat):
    ensure_chats_folder()
    chat_file = get_chat_file_path(chat["id"])

    with chat_file.open("w", encoding="utf-8") as file:
        json.dump(chat, file, indent=2)


# Delete one chat JSON file
def delete_chat_file(chat_id):
    chat_file = get_chat_file_path(chat_id)

    if chat_file.exists():
        chat_file.unlink()


# Set up session state for multiple chats
def initialize_session_state():
    if "chats" not in st.session_state:
        saved_chats = load_chats_from_disk()

        if saved_chats:
            st.session_state.chats = saved_chats
            st.session_state.active_chat_id = saved_chats[0]["id"]
        else:
            first_chat = create_chat()
            st.session_state.chats = [first_chat]
            st.session_state.active_chat_id = first_chat["id"]
            save_chat_to_disk(first_chat)
    elif "active_chat_id" not in st.session_state:
        if st.session_state.chats:
            st.session_state.active_chat_id = st.session_state.chats[0]["id"]
        else:
            st.session_state.active_chat_id = None


# Find the currently active chat
def get_active_chat():
    for chat in st.session_state.chats:
        if chat["id"] == st.session_state.active_chat_id:
            return chat
    return None


# Make a simple chat title from the first user message
def update_chat_title(chat, user_message):
    if chat["title"] == "New Chat":
        clean_title = user_message.strip()
        if clean_title:
            chat["title"] = clean_title[:30]
            save_chat_to_disk(chat)


# Remove a chat and switch to another one if needed
def delete_chat(chat_id):
    st.session_state.chats = [
        chat for chat in st.session_state.chats if chat["id"] != chat_id
    ]
    delete_chat_file(chat_id)

    if st.session_state.active_chat_id == chat_id:
        if st.session_state.chats:
            st.session_state.active_chat_id = st.session_state.chats[0]["id"]
        else:
            st.session_state.active_chat_id = None


# Build a system prompt from saved memory
def build_system_prompt():
    if not st.session_state.user_memory:
        return None

    memory_lines = []

    for key, value in st.session_state.user_memory.items():
        if isinstance(value, list):
            if value:
                memory_lines.append(f"{key}: {', '.join(str(item) for item in value)}")
        elif value:
            memory_lines.append(f"{key}: {value}")

    if not memory_lines:
        return None

    memory_text = "\n".join(memory_lines)
    return (
        "Use the following saved user memory to personalize your responses when it is helpful.\n"
        f"{memory_text}"
    )


# Add the memory system prompt before sending chat history to the model
def build_messages_for_api(messages):
    system_prompt = build_system_prompt()

    if system_prompt:
        return [{"role": "system", "content": system_prompt}] + messages

    return messages


# Function to send message history to the API
def send_message_to_api(messages):
    payload = {"model": model, "messages": build_messages_for_api(messages)}
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=request_timeout,
        )
        response.raise_for_status()
        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]

        st.error("Unexpected response format from the API.")
        return None
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            st.error("Invalid Hugging Face token. Please check your token.")
        elif response.status_code == 429:
            st.error("Rate limit exceeded. Please try again later.")
        else:
            st.error(f"HTTP error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None
    except ValueError as e:
        st.error(f"Error parsing response: {e}")
        return None


# Function to stream message history from the API
def stream_message_to_api(messages, response_placeholder):
    payload = {
        "model": model,
        "messages": build_messages_for_api(messages),
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    full_response = ""

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            stream=True,
            timeout=request_timeout,
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if not line.startswith("data: "):
                continue

            data_text = line[6:].strip()

            if data_text == "[DONE]":
                break

            try:
                data = json.loads(data_text)
            except json.JSONDecodeError:
                continue

            choices = data.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content", "")

            if content:
                full_response += content
                response_placeholder.markdown(full_response)
                time.sleep(0.02)

        if full_response:
            return full_response

        st.warning("Streaming returned no visible response, trying normal response mode.")
        return None
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            st.error("Invalid Hugging Face token. Please check your token.")
        elif response.status_code == 429:
            st.error("Rate limit exceeded. Please try again later.")
        else:
            st.error(f"HTTP error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None


# Parse a JSON object from a model response
def parse_json_object(text):
    clean_text = text.strip()

    if clean_text.startswith("```"):
        lines = clean_text.splitlines()
        if len(lines) >= 3:
            clean_text = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return {}

    return {}


# Merge newly extracted memory into saved memory
def merge_memory(existing_memory, new_memory):
    merged_memory = existing_memory.copy()

    for key, value in new_memory.items():
        if value in ("", None, [], {}):
            continue

        if isinstance(value, list):
            existing_list = merged_memory.get(key, [])
            if not isinstance(existing_list, list):
                existing_list = [existing_list] if existing_list else []

            for item in value:
                if item not in existing_list:
                    existing_list.append(item)

            merged_memory[key] = existing_list
        else:
            merged_memory[key] = value

    return merged_memory


# Make a second API call to extract user traits from the latest message
def extract_user_memory(user_message):
    extractor_messages = [
        {
            "role": "system",
            "content": (
                "Extract lasting user traits or preferences from the user's message. "
                "Return JSON only. Use only these keys when helpful: "
                "name, preferred_language, interests, communication_style, favorite_topics. "
                "Use strings or lists. Return {} if nothing useful is present."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    payload = {"model": model, "messages": extractor_messages}
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=request_timeout,
        )
        response.raise_for_status()
        data = response.json()

        if "choices" not in data or not data["choices"]:
            return {}

        content = data["choices"][0]["message"]["content"]
        return parse_json_object(content)
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError):
        return {}


initialize_session_state()
initialize_user_memory()

# Sidebar chat controls
with st.sidebar:
    st.header("Chats")

    if st.button("New Chat", use_container_width=True):
        new_chat = create_chat()
        st.session_state.chats.insert(0, new_chat)
        st.session_state.active_chat_id = new_chat["id"]
        save_chat_to_disk(new_chat)

    if st.session_state.chats:
        for chat in st.session_state.chats:
            button_col, delete_col = st.columns([4, 1])

            with button_col:
                if st.button(
                    chat["title"],
                    key=f"select_{chat['id']}",
                    use_container_width=True,
                    type="primary" if chat["id"] == st.session_state.active_chat_id else "secondary",
                ):
                    st.session_state.active_chat_id = chat["id"]

            with delete_col:
                if st.button("✕", key=f"delete_{chat['id']}", use_container_width=True):
                    delete_chat(chat["id"])
                    st.rerun()

            st.caption(chat["timestamp"])
    else:
        st.info("No chats yet. Click New Chat to start one.")

    with st.expander("User Memory", expanded=False):
        if st.session_state.user_memory:
            st.json(st.session_state.user_memory)
        else:
            st.write("No saved memory yet.")

        if st.button("Clear Memory", use_container_width=True):
            st.session_state.user_memory = {}
            save_memory_to_disk(st.session_state.user_memory)
            st.rerun()


active_chat = get_active_chat()

# Show the current chat in the main area
if active_chat:
    for message in active_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.info("No active chat. Create a new chat from the sidebar.")


# Chat input stays in the main area and sends messages to the active chat
prompt = st.chat_input(
    "Type your message here...",
    disabled=active_chat is None,
)

if prompt and active_chat:
    active_chat["messages"].append({"role": "user", "content": prompt})
    update_chat_title(active_chat, prompt)
    save_chat_to_disk(active_chat)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        ai_response = stream_message_to_api(active_chat["messages"], response_placeholder)

        if ai_response is None:
            ai_response = send_message_to_api(active_chat["messages"])
            if ai_response:
                response_placeholder.markdown(ai_response)

    if ai_response:
        active_chat["messages"].append({"role": "assistant", "content": ai_response})
        save_chat_to_disk(active_chat)

        new_memory = extract_user_memory(prompt)
        if new_memory:
            st.session_state.user_memory = merge_memory(
                st.session_state.user_memory,
                new_memory,
            )
            save_memory_to_disk(st.session_state.user_memory)
