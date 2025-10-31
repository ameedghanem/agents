from crewai.tools import BaseTool
from typing import Type
import os
from pydantic import BaseModel, Field
import requests


class PushNotificationInput(BaseModel):
    """A message to be sent to the user."""
    message: str = Field(..., description="The message to be sent to the user.")

class PushNotificationTool(BaseTool):
    name: str = "Send a push notification"
    description: str = (
        "This tool is used to send a push notification to the user."
    )
    args_schema: Type[BaseModel] = PushNotificationInput # its just a single argument, message !

    def _run(self, message: str) -> str:
        pushover_user = os.getenv("PUSHOVER_USER")
        pushover_toekn = os.getenv("PUSHOVER_TOKEN")
        pushover_url = "https://api.pushover.net/1/messages.json"
        
        print(f"Push: {message}")
        paylod = {"user": pushover_user, "token": pushover_toekn, "message": message}
        requests.post(pushover_url, data=paylod)
        return '{"notification": "ok"}'
