import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score

def main():
    st.title("Model Output Görselleştirici")
    st.write("Modelin Outputlarını Anlamlandırıp Sonuç Çıkarmamıza Yardımcı Olan Uygulama")
    
    uploaded_file = st.file_uploader("Bir JSON dosyası yükleyin:", type=["json"])
    
    if uploaded_file is not None:
        try:
            json_data = json.load(uploaded_file)
            
            if isinstance(json_data, list):
                df = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                df = pd.DataFrame([json_data])
            else:
                st.error("Geçerli bir JSON formatı içermiyor!")
                return

            targets = ["OTHER","SEXIST","RACIST","INSULT","PROFANITY"]
            
            if "target" in df.columns and "predict" in df.columns:
                y_true = df["target"]
                y_pred = df["predict"]

                st.subheader("CM")
                cm = confusion_matrix(y_true,y_pred,labels=targets)
                fig,ax = plt.subplots(figsize=(16,12))
                sns.heatmap(cm,annot=True,fmt="d",xticklabels=targets,yticklabels=targets)
                plt.xlabel("Tahmin Edilenler")
                plt.ylabel("Gerçek Değerler")
                st.pyplot(fig)

                st.subheader("Metrikler")
                acc = accuracy_score(y_true,y_pred)
                precision = precision_score(y_true,y_pred,average="weighted",zero_division=0)
                recall = recall_score(y_true,y_pred,average="weighted",zero_division=0)
                f1 = f1_score(y_true,y_pred,average="weighted",zero_division=0)

                st.write(f"ACCURACY : {acc:.4f}")
                st.write(f"PRECISION :  {precision:.4f}")
                st.write(f"RECALL :  {recall:.4f}")
                st.write(f"F1 :  {f1:.4f}")
            else:
                st.error("JSON dosyanız target ve predict içermiyor . Tekrar Deneyiniz !!")
        except json.JSONDecodeError:
            st.error("Geçersiz Format !!")

if __name__ == "__main__":
    main()
