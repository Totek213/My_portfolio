import openai
import speech_recognition as sr
import pyttsx3

# OpenAI API Configuration (Add your key)
openai.api_key = "YOUR_API_KEY"

# Initialize the speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed

# Function to speak
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio, language="en-US")
        print(f"Recognized: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("I couldn't understand, please repeat.")
        return None
    except sr.RequestError:
        print("Connection error.")
        return None

# Function to process the question using OpenAI
def ask_openai(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}]
        )
        answer = response["choices"][0]["message"]["content"]
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Main bot function
def voice_assistant():
    speak("Hello! How can I assist you?")
    while True:
        command = recognize_speech()
        if command:
            if "stop" in command or "exit" in command:
                speak("Goodbye!")
                break
            else:
                response = ask_openai(command)
                print("Assistant:", response)
                speak(response)

# Run the bot
if __name__ == "__main__":
    voice_assistant()
