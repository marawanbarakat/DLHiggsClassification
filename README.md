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

##How to add a new file

```
git add <path of you file>
```
##How to push your changes
```
git commit -am "new jupiter notebook"
git push m_DL localmaster
```  
##How to pull from Ece's master branch 
First add repo into your environment. This is a one time action.
```
git add ece_DL https://github.com/easilar/DLHiggsClassification.git
```
Now pull the changes
```
git pull ece_DL master
git commit -am "merge from ece's repository"
```
