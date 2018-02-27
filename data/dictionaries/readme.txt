[Inputs]

1. all2idx.pckl
- 5자리 진료코드, 3자리의 상병코드, 4자리의 처방코드 어느것을 넣어도 해당하는 idx를 0~1399로 반환

2. inputs.csv
- 0~1399에 해당하는 코드가 각각 어느 것인지 알기 쉽게 csv파일로 표시

3. d2i, s2i, p2i.pckl
- 진료, 상병, 처방의 코드를 넣으면 해당 idx 반환

4. sick_converter.pckl
- obsolete