# titanic-streamlit-app

This is a machine learning project that predicts the survival of passengers aboard the Titanic. It uses a trained logistic regression model, served via a Streamlit frontend, and can be deployed using Docker.

Model The Titanic survival prediction model is trained using the Titanic dataset. The model predicts whether a passenger survived based on various features such as:

Passenger Class (1st, 2nd, 3rd class)

Sex (male/female)

Age (age of the passenger)

Siblings/Spouses Aboard (sibsp)

Parents/Children Aboard (parch)

Fare (ticket fare)

Embarked (port of embarkation)

Features

Interactive Web Interface: Built using Streamlit to allow users to input passenger data and get survival predictions.

Model: The model is trained using Logistic Regression and saved as a .pkl file.

Dockerized: The app is containerized using Docker for easy deployment and distribution.
