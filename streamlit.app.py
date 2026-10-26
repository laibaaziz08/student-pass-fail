import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

# Set up the webpage.
st.set_page_config(
    page_title="Student Predictor",
    page_icon="🎓",
    layout="centered",
)

st.title("Student Pass / Fail Predictor")

st.write(
    "Enter the student's details below, "
    "then click Predict Result."
)

st.caption("A learning demo using a small example dataset.")

st.divider()

# Display two sliders side by side.
col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "Study Hours Per Day",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.5,
    )

with col2:
    prev_score = st.slider(
        "Previous Exam Score",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0,
    )

st.divider()

# Call FastAPI when the button is clicked.
if st.button("Predict Result"):
    data = {
        "study_hours": study_hours,
        "prev_score": prev_score,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()

            if result["prediction"] == 1:
                st.success(f"Prediction: {result['result_label']}")
            else:
                st.error(f"Prediction: {result['result_label']}")

            st.metric(
                "Model Confidence",
                f"{result['confidence'] * 100:.1f}%",
            )

            with st.expander("See the API input and response"):
                st.write("Input sent:")
                st.json(data)

                st.write("Response received:")
                st.json(result)

        else:
            st.error(
                f"The API returned an error: "
                f"{response.status_code}"
            )
            st.code(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. "
            "Start the prediction API in another terminal."
        )

    except requests.exceptions.Timeout:
        st.error("The request timed out. Please try again.")

    except requests.exceptions.RequestException as error:
        st.error(f"Request failed: {error}")

# Show the current slider values.
st.divider()
st.subheader("Current Input Summary")

left, right = st.columns(2)

left.metric("Study Hours", f"{study_hours} hrs/day")
right.metric("Previous Score", f"{prev_score}/100")