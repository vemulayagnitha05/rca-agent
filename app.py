import streamlit as st

from rca.schema import FailureRecord
from rca.systems.rule import RuleBased
from rca.taxonomy import DESCRIPTIONS, LABELS


st.set_page_config(
    page_title="RCA Agent",
    page_icon="??",
    layout="wide",
)

st.title("?? RCA Agent")
st.subheader("Root-cause analysis of software test failures")

st.info(
    "Offline demo: this interface uses the project's rule-based analyzer. "
    "It does not call an external LLM and does not claim research performance."
)

DEFAULT_LOG = """[INFO] Running test
java.lang.NullPointerException: Cannot invoke "String.length()" because value is null
    at com.acme.reportsvc.Invoice.resolve(Invoice.java:297)
    at com.acme.reportsvc.InvoiceTest.testResolve(InvoiceTest.java:25)
"""

col1, col2 = st.columns(2)

with col1:
    test_name = st.text_input(
        "Test name",
        value="InvoiceTest.testResolve",
    )

    commit_message = st.text_input(
        "Commit message",
        value="Refactor internals",
    )

    history_text = st.text_input(
        "Previous outcomes (optional)",
        value="P P P P",
        help="Use P for pass and F for failure, separated by spaces.",
    )

with col2:
    st.markdown("**Root-cause taxonomy**")
    for label in LABELS:
        st.caption(f"**{label}** — {DESCRIPTIONS[label]}")

log = st.text_area(
    "Paste the failing test log",
    value=DEFAULT_LOG,
    height=260,
)

diff = st.text_area(
    "Relevant code diff (optional)",
    value="diff --git a/src/Invoice.src b/src/Invoice.src\n+++ b/src/main/Invoice.java",
    height=120,
)

analyze = st.button("Analyze failure", type="primary", use_container_width=True)

if analyze:
    history = [x.strip().upper() for x in history_text.split() if x.strip()]
    history = [x for x in history if x in {"P", "F"}]

    record = FailureRecord(
        id="demo-case",
        source="streamlit-demo",
        project="demo",
        test_name=test_name,
        log=log,
        commit_message=commit_message,
        diff=diff,
        history=history,
    )

    diagnosis = RuleBased().diagnose(record)

    st.divider()
    st.markdown("## Diagnosis")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("Root cause", diagnosis.label)

    with result_col2:
        st.metric("Confidence", f"{diagnosis.confidence:.0%}")

    with result_col3:
        st.metric("Analyzer", "Rule-based")

    st.markdown("### Why?")
    st.write(diagnosis.hypothesis)

    st.markdown("### Evidence")
    if diagnosis.evidence:
        for evidence in diagnosis.evidence:
            st.code(evidence)
    else:
        st.warning("No strong evidence was found in the supplied log.")

    st.caption(
        "Demo only. The rule-based analyzer is one baseline in the research "
        "framework; demo output should not be interpreted as benchmark results."
    )

