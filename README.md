<div align="center">
  <h1>🍴 Charlotte Eatz 🤖</h1>
</div>

<div align="center"><b>Eric Phann</b>, product mgr. | <b>Yaxin Zhao</b>, data/prompt engr. | <b>Lakshmi Jayanth Kumar</b>, app dev. | <b>Gaurav Samdani</b>, LLM ops.</div>  
<br>

<div align="center">
  <a href="https://github.com/dsba6010-llm-applications/AgenticRAG-CharlotteEatz/blob/main/docs/Final%20Project%20Report.pdf">🎉 Final Report</a> |
  <a href="https://dinebot-uncc-dsba.streamlit.app/">🚀 Streamlit App</a>
</div>
<br>

<div align="center">
Welcome to <b>Charlotte Eatz</b>, your go-to app for exploring Charlotte's tastiest spots! We aim to simplify dining experiences in Charlotte, NC by seamlessly integrating restaurant reservations with additional end-to-end services like transportation booking and retrieving restaurant reviews. The core value and heart of Charlotte Eatz lies in its ability to streamline multiple tasks through <b>DineBot</b>, the app’s chatbot, saving time and improving users’ foodie experience in Charlotte.
</div>
<br>

<p align="center">
  <img width="500" height="500" src="https://raw.githubusercontent.com/dsba6010-llm-applications/AgenticRAG-CharlotteEatz/refs/heads/main/DineBot.png">
</p>

<p align="center"><i>DineBot, the heart of Charlotte Eatz. Generated by DALL-E.</i></p>
  
## ✨ Ready to Get Started?
### 1. Streamlit Cloud ☁️
The quickest and easiest way to our app is through [Streamlit Cloud!](https://dinebot-uncc-dsba.streamlit.app/)  
Simply input your __OpenAI API Key__ and you are ready to chat with DineBot 🤖!

### 2. Local Deployment ⚙️
First, clone the repo like this:

```bash
git clone --depth 1 https://github.com/dsba6010-llm-applications/AgenticRAG-CharlotteEatz.git
```

> [!WARNING]
> Our virtual environment was accidentally included in the initial push. It has since been removed from the repo but will be present in git history.
> Be sure to include ```--depth 1``` when cloning the repo to exclude git history and avoid downloading the virtual environment.

Then `cd` into the folder `AgenticRAG-CharlotteEatz`.  
Create a virtual environment, activate it, and install the dependencies.

```python
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> [!TIP]
> If you're using Windows CMD, the 2nd line will be `.\venv\Scripts\activate.bat`. 
> Alternatively, if you're using Windows PowerShell, it would be `.\venv\Scripts\activate.ps1`

## 🛠️ Behind the Scenes 
- **Core Architecture**: Powered by an **agentic RAG** (Retrieval-Augmented Generation) system for smarter interactions.
  - Built using **FAISS** for fast and efficient information retrieval.
  - Integrated with **LangChain** for agents, tools, and overall advanced conversational flows.



