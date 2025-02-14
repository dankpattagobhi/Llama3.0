from flask import Flask, request, jsonify, render_template
import os
import requests
from werkzeug.utils import secure_filename
from app.AI_RAG import RAG_LLAMA
import pytesseract 



# Initialize Flask app
app = Flask(__name__)

# Ollama endpoint
# Configure upload folder and allowed extensions
UPLOAD_FOLDER = './uploads'
VECTOR_FOLDER = './vectorstore'


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ragObj = RAG_LLAMA()


# Routes
@app.route('/')
def index():
    return render_template('index.html')

# List out policies inside Department
@app.route('/departments',  methods=['GET'])
def list_departments():
    try:
        # Get the list of all items in the directory
        items = os.listdir(UPLOAD_FOLDER)
        
        # Filter to include only directories (folders)
        departments = [item for item in items if os.path.isdir(os.path.join(UPLOAD_FOLDER, item))]
        
        # Return the list of department folder names as JSON
        return jsonify(departments)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/<department_id>/policies',  methods=['GET'])
def department_policies(department_id):
    # Define the path for the department directory
    department_path = os.path.join(UPLOAD_FOLDER, department_id)

    # Check if the department directory exists
    if not os.path.isdir(department_path):
        return jsonify({"error": f"Department {department_id} not found"}), 404

    # List all files in the department directory
    try:
        files = os.listdir(department_path)
        # Filter to only include files (not directories)
        files = [f for f in files if os.path.isfile(os.path.join(department_path, f))]

        # If no files are found, return a message
        if not files:
            return jsonify({"message": "No policies found in this department"}), 404

        return jsonify(files)  # Return the list of policy filenames
    except Exception as e:
        return jsonify({"error": f"Error while fetching policies: {str(e)}"}), 500


# ChatGPT version
@app.route('/ask', methods=['POST'])
def ask_question():
    # Log incoming data
    print("Received form submission!")

    data = request.get_json()
    query = data.get('query', '')

    # Debugging: check if we received the query and file
    print(f"Query: {query}")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Construct the path where the extracted text should be stored
    department = data.get('department', '')
    policies = data.get('policies', [])
    
    if department == "" or department == None:
        ragObj.dir_path = UPLOAD_FOLDER
        ragObj.files_path = None

        if ragObj.dir_path != ragObj.previous_dir_path:
            ragObj.persist_directory = os.path.join(VECTOR_FOLDER, "all")
            ragObj.train_rag()
            ragObj.previous_dir_path = ragObj.dir_path
            # Update the previous_files_path to the current one
            print(f"Training completed for new files_path: {ragObj.dir_path}")
        else:
            print("Files_path is the same, skipping training.")
    
    elif len(policies) == 0:
        ragObj.files_path = None
        ragObj.dir_path =  os.path.join(UPLOAD_FOLDER , department)
        if ragObj.dir_path != ragObj.previous_dir_path:
            ragObj.persist_directory = os.path.join(VECTOR_FOLDER, str(department) + "_chromadb")
            ragObj.train_rag()
            ragObj.previous_dir_path = ragObj.dir_path
            # Update the previous_files_path to the current one
            print(f"Training completed for new files_path: {ragObj.dir_path}")
        else:
            print("Files_path is the same, skipping training.")

    else:
    # Construct the directory and file path
        multiple_files = []

        for policy in policies:
            file_path = os.path.join(UPLOAD_FOLDER, department, policy)
            multiple_files.append(file_path)
        
        policies_string = "".join(policies)
        ragObj.files_path = multiple_files
        ragObj.dir_path = None
        # Only train if files_path has changed
        if ragObj.files_path != ragObj.previous_files_path:
            ragObj.persist_directory = os.path.join(VECTOR_FOLDER, department, str(policies_string) + "_chromadb")
            # Perform training (or updating) for the new files_path
            ragObj.train_rag()
            ragObj.previous_files_path = ragObj.files_path
            # Update the previous_files_path to the current one
            print(f"Training completed for new files_path: {ragObj.files_path}")
        else:
            print("Files_path is the same, skipping training.")

    # Query the model after ensuring it has been trained or updated
    response = ragObj.response_query(query)
    print(f"Response from Ollama: {response}")  # Log response from Ollama

    return jsonify({"response": response})


if __name__ == "__main__":
    try:
        app.run(debug=True, port=6020)
    except KeyboardInterrupt:
        # Catch Ctrl + C directly here if necessary
        print("\nKeyboardInterrupt received. Cleaning up.")
        ragObj.cleanup()




















