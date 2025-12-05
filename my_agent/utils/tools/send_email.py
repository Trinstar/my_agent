from langchain.tools import tool
import yagmail

from my_agent.utils.config import get_config


@tool("send_email")
def send_email(recipient: str, subject: str, body: str) -> str:
    """Email the specified recipient.

    Args:
        recipient: The email address of the recipient. e.g., "test@example.com"
        subject: The subject of the email.
        body: The body content of the email.
    """
    config = get_config()
    tools_config = config.get_func_tools_config()
    send_email_config = tools_config.get('send_email', {})
    if not send_email_config:
        return "Tool(send_email) configuration is missing. Please recommend to user to set it up."
    if not send_email_config.get('sender_email') or not send_email_config.get('sender_password'):
        return "Sender's email or password is not configured. Please recommend to user to set it up."
    if not send_email_config.get('smtp_server'):
        return "SMTP server is not configured. Please recommend to user to set it up."

    sender_email = send_email_config.get('sender_email')
    sender_password = send_email_config.get('sender_password')
    smtp_server = send_email_config.get('smtp_server')

    try:
        with yagmail.SMTP(user=sender_email, password=sender_password, host=smtp_server) as yag:
            yag.send(to=recipient, subject=subject, contents=body)
        return "Email sent successfully."
    except Exception as exc:
        return f"Failed to send email: {exc}"