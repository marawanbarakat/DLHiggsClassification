# DLHiggsClassification

Fork the repository using the Fork botton at the right top corner of this page

```
mkdir DLHiggsClassification
cd DLHiggsClassification
git init
git remote add m_DL https://github.com/marawanbarakat/DLHiggsClassification.git  
git fetch m_DL
git checkout -b localmaster m_DL/master
```

When uploading a new file, first save the file in the right directory in your local computer.
For example, if you are uploading a jupiter notebook save it in the jupiter notebook folder. 
Then follow the recipe below.

```
git add <path of you file>
git commit -am "new jupiter notebook"
git push m_DL localmaster
```  


