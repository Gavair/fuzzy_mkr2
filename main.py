import word
import words_model
import random
import lwa
import numpy as np
import streamlit as st


def main():
    st.title("Інтерфейс оцінки співробітника")
    st.header("Рівень навичок")

    low_skills = int(st.text_input("Низькі навички", value=2))
    basic_skills = int(st.text_input("Базові навички", value=2))
    intermediate_skills = int(st.text_input("Середні навички", value=2))
    proficient_skills = int(st.text_input("Високі навички", value=2))
    high_skills = int(st.text_input("Професійні навички", value=2))
    expert_skills = int(st.text_input("Експертні навички", value=2))

    grades = (
        ["Низькі навички"] * low_skills
        + ["Базові навички"] * basic_skills
        + ["Середні навички"] * intermediate_skills
        + ["Високі навички"] * proficient_skills
        + ["Професійні навички"] * high_skills
        + ["Експертні навички"] * expert_skills
    )

    model = words_model.words_skills

    W = []
    for item in model["words"]:
        W.append(grades.count(item))

    h = min(item["lmf"][-1] for item in model["words"].values())
    m = 50
    intervals_umf = lwa.alpha_cuts_intervals(m)
    intervals_lmf = lwa.alpha_cuts_intervals(m, h)

    res_lmf = lwa.y_lmf(intervals_lmf, model, W)
    res_umf = lwa.y_umf(intervals_umf, model, W)
    res = lwa.construct_dit2fs(
        np.arange(*model["x"]), intervals_lmf, res_lmf, intervals_umf, res_umf
    )

    sm = []
    model = words_model.words_skills
    for title, fou in model["words"].items():
        sm.append(
            (
                title,
                res.similarity_measure(
                    word.Word(None, model["x"], fou["lmf"], fou["umf"])
                ),
                word.Word(title, model["x"], fou["lmf"], fou["umf"]),
            ),
        )
    res_word = max(sm, key=lambda item: item[1])

    st.write(
        f'Рівень навичок співробітника: "{res_word[0]}", confidance: {round(res_word[1], 2)}'
    )
    res.plot()


if __name__ == "__main__":
    main()
