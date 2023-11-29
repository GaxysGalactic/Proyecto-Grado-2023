import asyncio
import os
import dqn_ai

from ai_flag import ai_flag

fwd_model_uri = os.environ.get(
    "FWD_MODEL_CONNECTION_STRING") or "ws://127.0.0.1:6969/?role=admin"

async def main():
    if ai_flag == "DQN":
        await dqn_ai.run_DQN(fwd_model_uri)
    else:
        print("NEAT")

if __name__ == "__main__":
    asyncio.run(main())
