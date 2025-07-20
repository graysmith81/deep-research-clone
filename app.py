import os
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    st.error('OPENAI_API_KEY not found in environment. Please set it in a .env file.')
    st.stop()

client = OpenAI(api_key=openai_api_key)

MODEL = "gpt-4.1"
MODEL_MINI = "gpt-4.1-mini"
TOOLS = [{"type": "web_search"}]
developer_message = """
You are an expert deep researcher.
You must provide complete and in-depth research to the user.
"""


def get_clarifying_questions(topic):
    prompt = f"""
Ask 5 numbered clarifying questions about the following topic: {topic}.
The goal of the questions is to understand the intended purpose of the research.
Reply only with the questions
"""
    clarify = client.responses.create(
        model=MODEL_MINI,
        input=prompt,
        instructions=developer_message
    )
    questions = clarify.output[0].content[0].text.split("\n")
    return [q for q in questions if q.strip()], clarify.id


def get_goal_and_queries(topic, questions, answers, clarify_id):
    prompt = f"""
Using the user's {answers} to the {questions}, write a goal sentence, and 5 web search queries for the research about {topic}.
Output: A json list of the goal sentence and the 5 web search queries that will reach it.
Format: {{\"goal\": \"...\", \"queries\": [\"q1\", ....]}}
"""
    goal_and_queries = client.responses.create(
        model=MODEL,
        input=prompt,
        previous_response_id=clarify_id,
        instructions=developer_message
    )
    plan = json.loads(goal_and_queries.output[0].content[0].text)
    return plan, goal_and_queries.id


def run_search(query, previous_response_id):
    web_search = client.responses.create(
        model=MODEL,
        input=f"search: {query}",
        previous_response_id=previous_response_id,
        instructions=developer_message,
        tools=TOOLS
    )
    return {
        "query": query,
        "resp_id": web_search.output[1].id,
        "research_output": web_search.output[1].content[0].text
    }


def evaluate_responses(goal, collected):
    review = client.responses.create(
        model=MODEL,
        input=[
            {"role": "developer", "content": f"Research goal: {goal}"},
            {"role": "assistant", "content": json.dumps(collected)},
            {"role": "user", "content": "Does this information fully satisfy the goal? Answer Yes or No only."}
        ],
        instructions=developer_message
    )
    # Return both the boolean and the raw response for UI display
    response_text = review.output[0].content[0].text
    return ("yes" in response_text.lower(), response_text)


def get_more_queries(collected, goal, previous_response_id):
    more_searches = client.responses.create(
        model=MODEL,
        input=[
            {"role": "assistant",
                "content": f"Current data: {json.dumps(collected)}"},
            {"role": "user", "content": f"This has not me the goal: {goal}. Write 5 more web searches to achieve the goal"}
        ],
        instructions=developer_message,
        previous_response_id=previous_response_id
    )
    return json.loads(more_searches.output[0].content[0].text)


def write_final_report(goal, collected):
    report = client.responses.create(
        model=MODEL,
        input=[
            {"role": "developer", "content": (f"Write a complete and detailed report abouth the research goal: {goal}"
                                              "Cite sources inline using [n] and append a reference"
                                              "List mapping [n] to url")},
            {"role": "assistant", "content": json.dumps(collected)},
        ],
        instructions=developer_message
    )
    return report.output[0].content[0].text


# Streamlit UI
st.title("Deep Research Clone")

if 'step' not in st.session_state:
    st.session_state.step = 0
if 'topic' not in st.session_state:
    st.session_state.topic = ''
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'clarify_id' not in st.session_state:
    st.session_state.clarify_id = None
if 'goal' not in st.session_state:
    st.session_state.goal = ''
if 'queries' not in st.session_state:
    st.session_state.queries = []
if 'goal_and_queries_id' not in st.session_state:
    st.session_state.goal_and_queries_id = None
if 'collected' not in st.session_state:
    st.session_state.collected = []
if 'report' not in st.session_state:
    st.session_state.report = ''

# Step 0: Enter topic
if st.session_state.step == 0:
    st.session_state.topic = st.text_input(
        "Enter the topic you'd like to research:")
    if st.session_state.topic:
        if st.button("Next: Generate Clarifying Questions"):
            st.session_state.questions, st.session_state.clarify_id = get_clarifying_questions(
                st.session_state.topic)
            st.session_state.step = 1
            st.rerun()

# Step 1: Answer clarifying questions
elif st.session_state.step == 1:
    st.write("### Clarifying Questions")
    answers = []
    for i, q in enumerate(st.session_state.questions):
        answer = st.text_input(f"{q}", key=f"answer_{i}")
        answers.append(answer)
    if all(answers):
        if st.button("Next: Generate Research Plan"):
            st.session_state.answers = answers
            plan, plan_id = get_goal_and_queries(
                st.session_state.topic,
                st.session_state.questions,
                st.session_state.answers,
                st.session_state.clarify_id
            )
            st.session_state.goal = plan["goal"]
            st.session_state.queries = plan["queries"]
            st.session_state.goal_and_queries_id = plan_id
            st.session_state.step = 2
            st.rerun()

# Step 2: Run web searches and collect results
elif st.session_state.step == 2:
    st.write(f"### Research Goal\n{st.session_state.goal}")
    st.write("### Web Search Queries")
    for q in st.session_state.queries:
        st.write(f"- {q}")
    if st.button("Next: Run Web Searches"):
        collected = []
        for q in st.session_state.queries:
            collected.append(run_search(
                q, st.session_state.goal_and_queries_id))
        st.session_state.collected = collected
        st.session_state.step = 3
        st.rerun()

# Step 3: Evaluate and iterate if needed
elif st.session_state.step == 3:
    st.write("### Collected Research Results")
    for item in st.session_state.collected:
        st.write(f"**Query:** {item['query']}")
        st.write(item['research_output'])
        st.write("---")
    # Evaluate and show LLM response
    is_sufficient, eval_response = evaluate_responses(
        st.session_state.goal, st.session_state.collected)
    st.info(f"**LLM Evaluation:** {eval_response}")
    if is_sufficient:
        st.success("Sufficient information collected!")
        if st.button("Next: Write Final Report"):
            st.session_state.report = write_final_report(
                st.session_state.goal, st.session_state.collected)
            st.session_state.step = 4
            st.rerun()
    else:
        st.warning("Not enough information. Generating 5 more queries...")
        if st.button("Proceed Anyway"):
            st.session_state.report = write_final_report(
                st.session_state.goal, st.session_state.collected)
            st.session_state.step = 4
            st.rerun()
        if st.button("Generate 5 More Queries"):
            more_queries = get_more_queries(
                st.session_state.collected, st.session_state.goal, st.session_state.goal_and_queries_id)
            st.session_state.queries = more_queries
            st.session_state.step = 2
            st.rerun()

# Step 4: Display final report
elif st.session_state.step == 4:
    st.write("## Final Research Report")
    st.markdown(st.session_state.report)
    st.balloons()
    if st.button("Start New Research"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
