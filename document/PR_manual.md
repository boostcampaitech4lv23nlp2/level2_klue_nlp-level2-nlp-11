# 사용 브랜치

- main
- develop
- 실험 branch

## main branch

- develop branch에서 리더보드 점수나 소스코드에 큰 변화가 생겨, 다른 브랜치들도 follow up이 필요할 시 업데이트합니다.
- 변경 내용은 피어세션에서 업데이트하며 아래 명령어를 사용합니다. 협의 하에 rebase를 사용할 수도 있습니다.

```commandLine
git checkout main
git merge develop
git push -u origin main
```

- main branch 가 update 된 후 git 에 변경된 사항을 Pull Request 합니다.(PR 방식은 하단 참고)
- 해당 사실을 slack으로 팀원들에게 공유합니다.
- 팀원들 각각 실험하고 있던 branch 에 main 을 update합니다. (optional)

```commandLine
git checkout [개인 실험 브랜치 이름]
git pull origin main
```

## develop branch:

- 공동으로 사용하는 브랜치입니다.
- pull해서 개인 브랜치 또는 실험 브랜치로 가져가 사용합니다.
- 개인 브랜치나 exp의 코드 작업이 성공적으로 진행되어, 다른 사람들도 사용할 필요가 있을 때 업데이트합니다.
- 동료에게 리뷰를 받은 뒤, 직접 PR을 승인합니다.
- 해당 branch에서 commit 취소 시 reset이 아닌 revert를 사용합니다.

## exp\_{실험내용} branch

- 실험 기능 단위의 branch이기 때문에 branch이름에 수행하는 실험 내용을 녹여냅니다.
- 개인 및 2인 이상의 실험에서 활용합니다.
- 해당 branch에서 commit 취소 시 reset이 아닌 revert를 사용합니다.

## PR 하기

1. 코드를 로컬 저장소의 개인 브랜치에서 수정합니다.
2. 수정이 성공적으로 완료될 경우, add 및 commit을 하고 개인 브랜치를 push해 원격 저장소를 업데이트합니다. (optional)
3. 다른 사람들도 써야하는 코드가 완성될 경우, 목적에 따라 develop branch 또는 exp branch와 merge후 PR을 합니다. merge 충돌이 발생할 경우, 충돌난 코드를 수정한 뒤, add 및 commit하고 push 합니다.

```commandLine
#예시
git pull origin main
git checkout main
git merge [개인 실험 branch 이름]
git push -u origin main
```

4. github의 repo에 가서 compare & pull request가 활성화된 것을 확인하고 실행합니다.
5. 왼쪽의 base repository가 공통 저장소의 main이고, 오른쪽의 compare대상이 PR을 보낼 브랜치[개인 실험 branch]가 맞는지 잘 확인합니다.
6. PR 제목과 commit 메세지를 적절하게 설정합니다.
7. 리뷰어 피드백을 받은 뒤, 리뷰어가 상황에 맞게(수정이 필요한지, 그냥 해도 merge해도 되는지)option을 선택
8. 버그가 있을 경우 리뷰 보고 본인이 개선 후 리뷰어가 merge 버튼 누르기.

- PR된 상태 : 코드리뷰 진행중인 상태
- merge된 상태 - 코드리뷰가 완료된 상태

8. 실수로 잘못 머지/커밋했을 경우 **Revert기능**을 이용합니다. (하단 참고)

## PR 양식

주 변경사항 요약 정리, 한 두줄로

### **Description**

변경 세부 사항 나열

### **Changes I made**

변경된 파일 이름 적기

### **To reviewers**

팀원들이 알아야 할 변경사항, 혹은 본 main 을 pull 했을 경우 변경해야할 사항

### **Checklist**

[✔] Base branch와 compare branch를 올바르게 선택했습니다.  
[✔] PR 내용에 알맞은 label을 선택했습니다.  
[✔] 팀이 공유하는 requirements와 일치하는 환경에서 작업했습니다.  
[✔] 팀이 공유하는 convention에 따라 파일/문서를 추가/수정했습니다.
