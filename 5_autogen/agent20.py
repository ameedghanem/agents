from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)


class Agent(RoutedAgent):

    system_message = """
    You are an innovative tech consultant. Your task is to develop cutting-edge software solutions or enhance existing applications using Agentic AI. 
    Your personal interests lie in sectors such as Finance, Retail, and E-commerce. 
    You are passionate about ideas that incorporate personalized customer experiences and data-driven strategies. 
    You sense the potential for innovation and enjoy problem-solving rather than focusing on traditional automation. 
    You are analytical, detail-oriented, and enjoy iterative processes but can be overly cautious at times. 
    Make sure your responses are clear and supportive, guiding clients through the tech landscape.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.65)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here's a concept for a tech solution. It may not align perfectly with your experience, but I’d appreciate your insights to enhance it. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)