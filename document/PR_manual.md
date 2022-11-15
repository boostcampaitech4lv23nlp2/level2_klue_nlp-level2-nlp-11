# 사용 브랜치

main 및 기능별 실험 branch

## main branch

- 기능별 실험 branch에서 main branch 에 update 시에 아래와 같이 입력합니다.

```commandLine
git checkout main
git merge [병합할 실험 branch 이름]
git push -u origin main
```

- main branch 가 update 된 후 git 에 변경된 사항을 Pull Request 합니다.(PR 방식은 하단 참고)
- 해당 사실을 slack으로 팀원들에게 공유합니다.
- 팀원들 각각 실험하고 있던 branch 에 main 을 update합니다.

```commandLine
git checkout [개인 실험 브랜치 이름]
git pull origin main
```

## exp\_{실험내용} branch

- 실험 기능 단위의 branch이기 때문에 branch이름에 수행하는 실험 내용을 녹여냅니다.
- 개인 및 2인 이상의 실험에서 활용합니다.
- 해당 branch에서 commit 취소 시 reset이 아닌 revert를 사용합니다.

## PR 하기

1. 코드를 로컬 저장소의 개인 브랜치에서 수정합니다.
2. 수정이 성공적으로 완료될 경우, add 및 commit을 하고 개인 브랜치를 push해 원격 저장소를 업데이트합니다. (optional)
3. 다른 사람들도 써야하는 코드가 완성될 경우, 목적에 따라 main branch 또는 exp branch와 merge후 PR을 합니다. merge 충돌이 발생할 경우, 충돌난 코드를 수정한 뒤, add 및 commit하고 push 합니다.

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
7. 동료의 피드백을 받은 뒤 , PR을 보낸 본인이 github Pull requests에서 직접 merge를 합니다.

- PR된 상태 : 코드리뷰 진행중인 상태
- merge된 상태 - 코드리뷰가 완료된 상태

8. 실수로 잘못 머지/커밋했을 경우 **Revert기능**을 이용합니다. (하단 참고)
