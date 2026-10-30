import asyncio
import os

import streamlit as st
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Model configuration
MODEL_NAME = os.getenv(
    "LOCAL_LLM_MODEL",
    "ai/qwen3-vl:2B-BF16",
)

BASE_URL = os.getenv(
    "LOCAL_LLM_BASE_URL",
    "http://localhost:12434/v1",
)


# Streamlit configuration
st.set_page_config(
    page_title="AI Job Description Generator",
    page_icon="💼",
    layout="centered",
)

st.title("💼 AI Job Description Generator")
st.write(
    "Enter a job title and let your AI HR assistant "
    "generate a professional job description."
)


async def generate_job_description(job_title: str) -> str:
    """Generate a job description using AutoGen and a local LLM."""

    model_client = OpenAIChatCompletionClient(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key="local-model",
        model_info={
            "vision": True,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
        },
    )

    try:
        hr_agent = AssistantAgent(
            name="hr_assistant",
            model_client=model_client,
            system_message=(
                "You are an HR assistant. "
                "You specialize in writing professional job descriptions. "
                "Given a job title, generate a complete and relevant job "
                "description suitable for a job posting. "
                "Include the following sections: "
                "Job Summary, Key Responsibilities, Required Skills, "
                "and Qualifications. "
                "Use clear headings and bullet points. "
                "Keep the description concise, realistic, and professional. "
                "Do not invent company-specific information or salary figures. "
                "Return only the generated job description."
            ),
        )

        task = (
            f"Generate a professional job description for the following role: "
            f"{job_title}. "
            "Include Job Summary, Key Responsibilities, Required Skills, "
            "and Qualifications."
        )

        result = await hr_agent.run(task=task)

        for message in reversed(result.messages):
            if (
                getattr(message, "source", None) == "hr_assistant"
                and isinstance(getattr(message, "content", None), str)
            ):
                response = message.content.strip()
                if response:
                    return response

        raise RuntimeError("The model returned an empty response.")

    finally:
        await model_client.close()


# User input: job title only
job_title = st.text_input(
    "Enter Job Title",
    placeholder="e.g., Junior Python Developer",
)

if st.button("Generate Job Description", type="primary"):
    if not job_title.strip():
        st.warning("Please enter a job title.")
    else:
        with st.spinner("AI HR Assistant is generating your job description..."):
            try:
                description = asyncio.run(
                    generate_job_description(job_title.strip())
                )

                st.success("Job description generated successfully!")

                st.subheader(job_title.strip())
                st.markdown(description)

                st.download_button(
                    label="Download Job Description",
                    data=description,
                    file_name="job_description.md",
                    mime="text/markdown",
                )

            except Exception as exc:
                st.error(
                    "Failed to generate the job description. "
                    "Check whether your local model server is running "
                    "and the model API configuration is correct."
                )

                with st.expander("Technical error details"):
                    st.code(f"{type(exc).__name__}: {exc}")


st.divider()
st.caption("Powered by Streamlit, Microsoft AutoGen, and your local Qwen model.")