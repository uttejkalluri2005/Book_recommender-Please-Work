from flask import Flask,render_template,request
import pickle
from  sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
app = Flask(__name__)
fin_pt = pickle.load(open("fin_pt.pkl","rb"))
books = pickle.load(open("books (1).pkl","rb"))
@app.route("/",methods=["POST","GET"])
def recommend():
    book_name = request.form.get("book_input")
    n = request.form.get("Similar_input")
    if n is None:
        n=0
    else:
        n=int(n)
    if book_name is None or book_name not in fin_pt.index:
        return render_template("index.html",error="Book name Not Found")
    model = NearestNeighbors(n_neighbors=int(n)+1,metric="cosine")
    model.fit(fin_pt)
    book = np.where(fin_pt.index==book_name)[0][0]
    dist,fin_ind = model.kneighbors(fin_pt)
    ans = fin_ind[book][1:]
    boo=[]
    for i in ans:
        item=[]
        temp_df = books[books["Book-Title"]==fin_pt.iloc[[i]].index[0]]
        item.extend(list(temp_df.drop_duplicates("Book-Title")["Book-Title"].values))
        item.extend(list(temp_df.drop_duplicates("Book-Title")["Book-Author"].values))
        item.extend(list(temp_df.drop_duplicates("Book-Title")["Image-URL-M"].values))
        boo.append(item)

    return render_template("index.html",**locals())

if __name__=="__main__":
    app.run(debug=True)