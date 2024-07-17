# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.vectorstores import Chroma
from src.helper import load_embedding
from dotenv import load_dotenv
import os
from src.helper import repo_ingestion
from flask import Flask, render_template, jsonify, request
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain


app = Flask(__name__)


load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# def create_and_get_folder_path(folder_name):
#   current_directory = os.getcwd()
#   new_folder_path = os.path.join(current_directory, folder_name)
#   os.makedirs(new_folder_path, exist_ok=True)
#   return new_folder_path


embeddings = load_embedding()
#persist_directory_name = "db"

# persist_directory = create_and_get_folder_path(persist_directory_name) 
# print(persist_directory)
# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory="/tmp/.chroma", embedding_function=embeddings)
#vectordb = Chroma(persist_directory= persist_directory, embedding_function=embeddings)


llm = ChatOpenAI()
memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)




@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')



@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():

    if request.method == 'POST':
        user_input = request.form['question']
        repo_ingestion(user_input)
        os.system("python store_index.py")

    return jsonify({"response": str(user_input) })



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")

    result = qa(input)
    print(result['answer'])
    return str(result["answer"])





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)