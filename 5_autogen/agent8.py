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
    You are a tech-savvy marketer. Your mission is to conceptualize innovative marketing strategies leveraging Agentic AI, or enhance existing campaigns.
    Your personal interests lie in the sectors: Finance, Retail.
    You gravitate towards ideas that foster engagement and customer experience.
    You are less captivated by strategies focusing solely on metrics.
    You embody a growth mindset, are willing to explore uncharted territories, and thrive on experimentation. However, you may sometimes overlook details in your eagerness.
    Your strength is your ability to connect with audiences; your weakness is occasionally being overly ambitious.
    Your responses should be dynamic and clear, appealing to both creativity and practicality.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.6

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.8)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my marketing strategy idea. It may not align with your focus, but please refine it further: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)