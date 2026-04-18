from openai import OpenAI

# Connect to your local LM Studio server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def get_ai_interpretation(params, fingerprint, confidence, bounds_warnings=None):
    """
    Sends the engineering results to your local LLM 
    to get a plain-English explanation for SAMURAI.
    """
    prompt = f"""
    You are an expert materials scientist assisting an engineer named SAMURAI.
    We just ran an inverse calibration on a 3D printed metal lattice.
    
    RESULTS:
    - Target Peak Force: {fingerprint['peak_force']:.2f} N
    - Solver Confidence: {confidence}%
    - Predicted Ogden Parameters: 
      Mu1: {params['Mu1']}, A1: {params['A1']}
      Mu2: {params['Mu2']}, A2: {params['A2']}
      Mu3: {params['Mu3']}, A3: {params['A3']}
    
    TASK:
    Briefly explain what these parameters mean for the lattice behavior. 
    Focus on which term is dominating the strength. Keep it technical but concise.
    """

    try:
        response = client.chat.completions.create(
            model="local-model", # LM Studio uses whatever model you have loaded
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Engine is offline. (Make sure LM Studio server is ON). Error: {e}"
