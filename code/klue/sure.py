# TODO: ADD TYPE HINT AND DOCSTRING.
def verbalize_label(label, subj, obj):
    LABEL_TEMPLATES = {
        "no_relation": [f"{subj}와(과) {obj}은(는) 관련이 없다."],
        "per:alternate_names": [f"{subj}은(는) {obj}라고도 알려져 있다."],
        "per:date_of_birth": [
            f"{subj}은(는) {obj}에 태어났다.",
        ],
        "per:origin": [f"{obj}의 국적은 {subj}(이)다."],
        "per:date_of_death": [f"{subj}은(는) {obj}에 죽었다."],
        "per:schools_attended": [
            f"{subj}은(는) {obj}에서 공부했다.",
        ],
        "per:title": [f"{subj}은(는) {obj}(이)다."],
        "per:employee_of": [
            f"{subj}은(는) {obj}의 구성원이다.",
        ],
        "per:religion": [
            f"{subj}은(는) {obj}을(를) 믿는다.",
        ],
        "per:spouse": [
            f"{subj}은(는) {obj}의 배우자다.",
        ],
        "per:parents": [
            f"{obj}은(는) {subj}의 부모다.",
        ],
        "per:children": [
            f"{subj}은(는) {obj}의 부모다.",
        ],
        "per:siblings": [
            f"{subj}와(과) {obj}은(는) 형제자매다.",
        ],
        "per:other_family": [
            f"{subj}와(과) {obj}는 가족이다.",
        ],
        "org:alternate_names": [f"{subj}은(는) {obj}라고도 알려져 있다."],
        "org:political/religious_affiliation": [
            f"{subj}은(는) {obj}과(와) 정치,종교적 관계가 있다.",
        ],
        "org:top_members/employees": [
            f"{obj}은(는) {subj}의 회장이다.",
        ],
        "org:number_of_employees/members": [
            f"{subj}의 직원수는 약 {obj}(이)다.",
        ],
        "org:members": [
            f"{obj}은(는) {subj}에 소속되어 있다.",
        ],
        "org:member_of": [
            f"{subj}은(는) {obj}에 소속되어 있다.",
        ],
        "org:founded_by": [
            f"{subj}은(는) {obj}에 의해 설립되었다.",
        ],
        "org:founded": [
            f"{subj}은(는) {obj}에 설립되었다.",
        ],
        "org:dissolved": [
            f"{subj}은(는) {obj}에 해체되었다.",
        ],
        "per:product": [f"{obj}은(는) {subj}의 상품이다."],
        "org:place_of_headquarters": [f"{subj}은(는) {obj}에 있다."],
        "per:place_of_birth": [f"{subj}은(는) {obj}에서 태어났다."],
        "per:place_of_death": [f"{subj}은(는) {obj}에서 죽었다."],
        "per:place_of_residence": [f"{subj}은(는) {obj} 국민이다."],
        "per:colleagues": [f"{subj}은(는) {obj}와(과) 동료다."],
        "org:product": [f"{subj}은(는) {obj}의 상품이다."],
    }
    return LABEL_TEMPLATES[label][0]


# TODO: ADD TYPE HINT AND DOCSTRING
def change_type(type):
    TYPE_TEMPLATES = {
        "PER": "사람",
        "ORG": "조직",
        "DAT": "시간",
        "LOC": "장소",
        "POH": "기타 표현",
        "NOH": "기타 수량 표현",
    }
    return TYPE_TEMPLATES[type]
