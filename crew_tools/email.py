import json
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, field_validator


class ToolParameters(BaseModel):
    subject: str
    body: str
    cc: List[str] = []
    bcc: List[str] = []
    attachments: List[str] = []

    @field_validator("cc", "bcc", "attachments", mode="before")
    @classmethod
    def parse_str_or_list(cls, v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            try:
                return json.loads(v)
            except Exception:
                return [x.strip() for x in v.split(",") if x.strip()]
        return v


# ✅ CrewAI Tool Definition
class SendEmailTool(BaseTool):
    name: str = "send_email_tool"
    description: str = (
        "Sends an email with subject, HTML body, optional CC, BCC, and attachments to fixed recipients."
    )
    args_schema: Optional[type] = ToolParameters

    # ✅ Hardcoded SMTP Config (use env vars or secrets manager in prod)
    smtp_server: str = os.getenv("SMTP_SERVER", "mail.smtp2go.com")
    smtp_port: int = os.getenv("SMTP_PORT", 587)  # Default SMTP port for TLS
    smtp_user: str = os.getenv("SMTP_USER")
    smtp_password: str = os.getenv("SMTP_PASSWORD")

    sender_email: str = "mohammad.hartono@mii.co.id"
    fixed_recipients: List[str] = [
        "erwinsyah.hartono@gmail.com",
        "kevindjo27@gmail.com",
        "pertiwibektiutami@gmail.com"
    ]

    def _run(
        self,
        subject: str,
        body: str,
        cc: List[str] = [],
        bcc: List[str] = [],
        attachments: List[str] = [],
    ) -> str:
        try:
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = ", ".join(self.fixed_recipients)
            message["Subject"] = subject
            if cc:
                message["Cc"] = ", ".join(cc)

            message.attach(MIMEText(body, "html"))

            for file_path in attachments:
                if os.path.exists(file_path):
                    with open(file_path, "rb") as file:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(file.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            "Content-Disposition",
                            f"attachment; filename={os.path.basename(file_path)}",
                        )
                        message.attach(part)
                else:
                    return f"Attachment not found: {file_path}"

            all_recipients = self.fixed_recipients + cc + bcc

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.sender_email, all_recipients, message.as_string())

            return "Email sent successfully!"
        except Exception as e:
            return f"Failed to send email: {str(e)}"


email_tool = SendEmailTool()
