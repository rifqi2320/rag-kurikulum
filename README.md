## How to run preprocess data
0. Make sure vector store is running and the env is configured
1. Change to app directory and run the following command
```bash
python load_documents.py
```

## How to install and run
0. Run docker compose for pgvector
```bash
docker-compose up -d
```
1. Create a virtual environment
```bash
python3 -m venv venv
```
2. Activate the virtual environment
```bash
source venv/bin/activate
```
3. Install the requirements
```bash
pip install -r requirements.txt
```
4. Run the application
```bash
python app/main.py
```
4. (Optional) Run dev

Change directory to app  
```bash
cd app
```
Run the following command  
```bash
uvicorn main:app --reload
```

## Note:
- If the prompt is not complex, use english as the system prompts, and give examples in other language (https://arxiv.org/pdf/2304.05613)