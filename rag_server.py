import os
import requests
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# =========================================================================
# CONFIGURATION
# =========================================================================
# Paste your TBA read API key here
TBA_API_KEY = "jca8IBbkSBd4NGP01K2AN6MYH4HAkI4FBdVIDZrUJnFsxsYiinN5RZiOPhYNLIUL" 

app = FastAPI()

# Enable CORS so your local HTML file can securely fetch data from this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize a persistent local vector database directory
db_client = chromadb.PersistentClient(path="./chroma_db")
# Using Chroma's default built-in embedding model (no internet needed to vectorize!)
collection = db_client.get_or_create_collection("frc_match_history")

# =========================================================================
# PHASE 1 & 2: SCRAPE AND INDEX TBA DATA
# =========================================================================
@app.get("/api/init-event")
def init_event(event_key: str):
    """
    Scrapes the qualification match data for an event from TBA,
    generates structured historical context text strings, and embeds them into ChromaDB.
    """
    if not TBA_API_KEY or TBA_API_KEY == "YOUR_TBA_API_KEY_HERE":
        return {"status": "error", "message": "Please configure your TBA_API_KEY inside rag_server.py first!"}

    url = f"https://www.thebluealliance.com/api/v3/event/{event_key}/matches"
    headers = {"X-TBA-Auth-Key": TBA_API_KEY}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return {"status": "error", "message": f"TBA API returned status code {response.status_code}"}
        
        matches = response.json()
        qual_matches = [m for m in matches if m.get("comp_level") == "qm"]
        
        if not qual_matches:
            return {"status": "error", "message": f"No qualification matches found for event key {event_key}."}
        
        documents = []
        ids = []
        metadatas = []
        
        for m in qual_matches:
            match_num = m.get("match_number")
            blue_all = [t.replace("frc", "") for t in m["alliances"]["blue"]["team_keys"]]
            red_all = [t.replace("frc", "") for t in m["alliances"]["red"]["team_keys"]]
            
            # Extract basic scoring breakdown details if matches have already been played
            score_breakdown = m.get("score_breakdown")
            breakdown_text = ""
            if score_breakdown:
                b_score = score_breakdown["blue"].get("totalPoints", 0)
                r_score = score_breakdown["red"].get("totalPoints", 0)
                breakdown_text = f"Result: Blue scored {b_score}, Red scored {r_score}."
            
            # Construct a human-readable historical match context text block
            context_str = (
                f"In match {match_num} at event {event_key}, the Blue alliance consisted of teams {', '.join(blue_all)} "
                f"and the Red alliance consisted of teams {', '.join(red_all)}. {breakdown_text}"
            )
            
            # Unique ID structure to prevent data duplication inside ChromaDB
            doc_id = f"{event_key}_qm_{match_num}"
            
            documents.append(context_str)
            ids.append(doc_id)
            metadatas.append({"event": event_key, "match_number": match_num})
        
        # Upsert (insert or update) the data blocks directly into our vector database
        collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
        return {"status": "success", "message": f"Successfully indexed {len(documents)} matches for event {event_key}."}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

# =========================================================================
# PHASE 3: CONTEXT RETRIEVAL ENDPOINT
# =========================================================================
class StrategyRequest(BaseModel):
    blue_teams: list
    red_teams: list

@app.post("/api/rag-strategy")
def get_rag_strategy(req: StrategyRequest):
    """
    Queries ChromaDB to extract historical tactical overlaps for the requested alliance match.
    """
    all_teams = req.blue_teams + req.red_teams
    query_string = f"Teams playing: {', '.join(all_teams)}"
    
    # Query our database to find the top 5 matches with the highest mathematical text similarity
    results = collection.query(
        query_texts=[query_string],
        n_results=min(5, collection.count()) if collection.count() > 0 else 0
    )
    
    retrieved_context = []
    if results and results.get("documents") and len(results["documents"]) > 0:
        retrieved_context = results["documents"][0]
        
    return {
        "context": retrieved_context
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)