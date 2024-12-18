import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import firebase_admin
from firebase_admin import credentials, firestore
import uuid

def main():
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase_config.json")
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()

    st.title("Model Output Görselleştirme")
    st.write("Modelin Outputlarını Anlamlandırıp Sonuç Çıkarmamıza Yardımcı Olan Uygulama")
     
    menu = st.sidebar.selectbox("Sayfalar", ["Model Output Yükle", "Dashboard"])
    
    targets = ["OTHER", "SEXIST", "RACIST", "INSULT", "PROFANITY"]

    if menu == "Model Output Yükle":
        st.subheader("Model Output Yükle")

        model_name = st.text_input("Model Adı :")
        uploaded_file = st.file_uploader("Output Dosyasını JSON yükle :", type=["json"])
        save_button = st.button("Kaydet")

        if uploaded_file is not None:
            try:
                json_data = json.load(uploaded_file)

                if all(key in json_data for key in ["train", "test", "valid"]):
                    metrics = {}
                    for subset in ["train", "test", "valid"]:
                        data = json_data[subset]
                        if isinstance(data, list) and all("target" in item and "predict" in item for item in data):
                            df = pd.DataFrame(data)
                            y_true = df["target"]
                            y_pred = df["predict"]

                            cm = confusion_matrix(y_true, y_pred, labels=targets)
                            acc = accuracy_score(y_true, y_pred)
                            precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                            recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

                            metrics[subset] = {
                                "cm": cm,
                                "accuracy": acc,
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                            }
                        else:
                            st.warning(f"{subset.capitalize()} verisi uygun formatta değil. 'target' ve 'predict' içermeli.")
                else:
                    st.error("JSON dosyasında 'train', 'test' ve 'valid' anahtarları eksik!")
            except json.JSONDecodeError:
                st.error("Geçersiz JSON formatı!")

        if save_button:
            if not model_name:
                st.warning("Lütfen bir model adı girin.")
            elif uploaded_file is None:
                st.warning("Lütfen bir JSON dosyası yükleyin.")
            else:
                existing_model_query = db.collection("models").where("model_name", "==", model_name).stream()
                existing_model_docs = list(existing_model_query)

                if existing_model_docs:
                    st.error(f"{model_name} adlı model zaten mevcut!")

                elif 'metrics' in locals() and metrics:
                    model_data = {
                        "model_name": model_name,
                        "results": {
                            subset: {
                                "cm": metrics[subset]["cm"].tolist(),
                                "accuracy": metrics[subset]["accuracy"],
                                "precision": metrics[subset]["precision"],
                                "recall": metrics[subset]["recall"],
                                "f1": metrics[subset]["f1"],
                            }
                            for subset in metrics
                        }
                    }
                    save_to_firestore(db, model_name, subset,cm, acc, precision, recall, f1)
                    st.success(f"{model_name} modeli başarıyla kaydedildi!")
                else:
                    st.error("Veriler işlenirken bir hata oluştu!")


    elif menu == "Dashboard":
        st.subheader("Dashboard")
    
        models = db.collection("models").stream()
        model_list = [{"id": model.id, **model.to_dict()} for model in models]

        if model_list:
            model_names = [model["model_name"] for model in model_list]
            selected_model = st.selectbox("Bir model seçin:", model_names)

            if selected_model:
                selected_model_data = next(model for model in model_list if model["model_name"] == selected_model)
                st.write(f"Seçilen Model: **{selected_model_data['model_name']}**")

                results = selected_model_data.get("results", {})
                for subset, metrics in results.items():
                    cm = [metrics["cm"][key] for key in sorted(metrics["cm"].keys())]
                    acc = metrics["accuracy"]
                    precision = metrics["precision"]
                    recall = metrics["recall"]
                    f1 = metrics["f1"]

                    st.subheader(f"{subset.capitalize()} Metrikleri")
                    handle_metrics_show(cm, targets, acc, precision, recall, f1)
        else:
            st.warning("Henüz kayıtlı model bulunmuyor!")


def handle_metrics_show(cm, targets, acc, precision, recall, f1):
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=targets, yticklabels=targets, cmap="Blues")
    plt.xlabel("Tahmin Edilenler")
    plt.ylabel("Gerçek Değerler")
    st.pyplot(fig)

    st.subheader("Metrikler")
    st.write(f"ACCURACY : {acc:.4f}")
    st.write(f"PRECISION : {precision:.4f}")
    st.write(f"RECALL : {recall:.4f}")
    st.write(f"F1 : {f1:.4f}")

def save_to_firestore(db, model_name, subset, cm, acc, precision, recall, f1):
    models_ref = db.collection("models")
    
    cm_list = cm.tolist()
    cm_dict = {f"row_{i}": row for i, row in enumerate(cm_list)}

    model_data = {
        "model_name": model_name,
        "results": {
            subset: {
                "cm": cm_dict,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        }
    }

    existing_model_query = models_ref.where("model_name", "==", model_name).stream()
    existing_model_docs = list(existing_model_query)

    if existing_model_docs:
        model_doc_ref = existing_model_docs[0].reference
        model_doc_ref.update({
            f"results.{subset}": model_data["results"][subset]
        })
    else:
        model_doc_ref = models_ref.add(model_data)[1]

    return model_doc_ref


if __name__ == "__main__":
    main()
