import joblib
import streamlit as st
import pandas as pd
import numpy as np

def load_recommendation_model(cuisine, avg_cost, rating):
    rec_dataset = pd.read_csv('recommendation_dataset.csv')  
    tfidf = joblib.load('tfidf_vectorizer.joblib')  
    nn = joblib.load('recommendation_model.joblib')             

    filtered = rec_dataset[
        (rec_dataset['Cuisines'].str.lower().str.contains(cuisine.lower())) &
        (rec_dataset['Average Cost for two'] >= avg_cost) &
        (rec_dataset['Aggregate rating'] >= rating)
    ].reset_index(drop=True)

    if not filtered.empty:
        reference = filtered.iloc[0]
        ref_tfidf = tfidf.transform([reference['Cuisines']])
        ref_features = np.hstack((
            ref_tfidf.toarray(),
            np.array([[reference['Average Cost for two'], reference['Price range'], reference['Aggregate rating']]])
        ))
        distances, indices = nn.kneighbors(ref_features, n_neighbors=5)

        recommendations = []
        for idx in indices[0]:
            rec = rec_dataset.iloc[idx]
            recommendations.append(f"{rec['Restaurant Name']} ({rec['Cuisines']})")
        return recommendations
    else:
        return ["No recommendations found matching your preferences."]

def main():
    st.title("Restaurant Recommendation System")
    st.write("Get personalized restaurant recommendations based on cuisine, cost, and rating.")

    cuisine = st.text_input("Enter preferred cuisine:")
    avg_cost = st.number_input("Enter minimum average cost for two:")
    rating = st.number_input("Enter minimum rating:", min_value=0.0, max_value=5.0)

    if st.button("Get Recommendations"):
        if cuisine and avg_cost > 0 and rating >= 0:
            recommendations = load_recommendation_model(cuisine, avg_cost, rating)
            st.success("Recommended Restaurants:")
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.error("Please fill all fields correctly.")

if __name__ == "__main__":
    main()