## domain_norms
1. **RuleID: NO_0_SEMANTIC** (Hard)  
   **描述**: "no"表示不存在/未应用，"0"表示数值零，二者不可互换。  
   **触发条件**: 任何列中出现"no"与"0"的语义冲突（如"HTL"列出现"0"）。  
   **违反处理**: Reject  

2. **RuleID: TEMP_RANGE** (Hard)  
   **描述**: 热处理温度必须符合实验合理范围（50-300°C）。  
   **触发条件**: "Annealing temperature (°C)"列出现"MS"或超出范围值。  
   **违反处理**: Repair（替换为"na"）  

3. **RuleID: ANNEALING_AMBIENCE** (Hard)  
   **描述**: 热处理环境若为"na"，则"Annealing time"必须为"na"。  
   **触发条件**: "Annealing ambience"为"na"但"Annealing time"非"na"。  
   **违反处理**: Flag  

4. **RuleID: HTL_DEPENDENCY** (Hard)  
   **描述**: 若"HTL"为"no"，则"HTL additive"必须为"no"。  
   **触发条件**: "HTL"为"no"但"HTL additive"非"no"。  
   **违反处理**: Repair  

5. **RuleID: DEPOSITION_METHOD** (Soft)  
   **描述**: "Perovskite deposition method"为"vasp"时，"Perovskite deposition procedure"必须为"two step"。  
   **触发条件**: "vasp"与"one step"共现。  
   **违反处理**: Flag  

6. **RuleID: ANTI_SOLVENT** (Hard)  
   **描述**: "Anti solvent treatment"为"no"时，"Precursor solution"不得为"DMF"。  
   **触发条件**: "no"与"DMF"共现。  
   **违反处理**: Repair  

## entity_description
1. **Article**  
   - 简介：文献引用格式  
   - 数据类型：字符串  
   - 缺失/空值：无  
   - 取值语义：作者名+期刊名+卷号+页码+年份  
   - 依赖/互斥：无  
   - 允许操作：字符串拼接  

2. **ETL**  
   - 简介：电子传输层材料  
   - 数据类型：字符串  
   - 缺失/空值：无  
   - 取值语义：TiO2
   - 依赖/互斥：无  
   - 允许操作：固定值复用  

3. **ETL 2**  
   - 简介：第二电子传输层材料  
   - 数据类型：字符串  
   - 缺失/空值：空字符串表示缺失  
   - 取值语义："mAl2O3"/"mTiO2"（材料）或"0"（未应用）  
   - 依赖/互斥：与"ETL"形成叠层结构  
   - 允许操作：分箱复用  

4. **Perovskite**  
   - 简介：钙钛矿材料组成  
   - 数据类型：字符串  
   - 缺失/空值：无  
   - 取值语义：MAPbI3/MAPbI3 xClx（含氯/不含氯）  
   - 依赖/互斥：与"Deposition procedure"关联  
   - 允许操作：枚举值复用  

5. **Perovskite deposition procedure**  
   - 简介：钙钛矿沉积工艺步骤  
   - 数据类型：字符串  
   - 缺失/空值：无  
   - 取值语义："one step"/"two step"  
   - 依赖/互斥：与"Deposition method"关联  
   - 允许操作：枚举值复用  

6. **Perovskite deposition method**  
   - 简介：钙钛矿沉积方法  
   - 数据类型：字符串  
   - 缺失/空值：无  
   - 取值语义："spin"/"vasp"  
   - 依赖/互斥：与"Deposition procedure"关联  
   - 允许操作：枚举值复用  

7. **Anti solvent treatment**  
   - 简介：抗溶剂处理  
   - 数据类型：字符串  
   - 缺失/空值：无  
   - 取值语义："no"（未应用）  
   - 依赖/互斥：与"Precursor solution"关联  
   - 允许操作：固定值复用  

8. **Precursor solution**  
   - 简介：前驱体溶液  
   - 数据类型：字符串  
   - 缺失/空值：无  
   - 取值语义：DMF 
   - 依赖/互斥：与"Anti solvent treatment"关联  
   - 允许操作：固定值复用  

9. **Annealing temperature (°C) **  
   - 简介：退火温度  
   - 数据类型：字符串  
   - 缺失/空值："na"表示缺失  
   - 取值语义：50-300°C范围值或"MS"（未指定）  
   - 依赖/互斥：与"Annealing ambience"关联  
   - 允许操作：分箱复用  

10. **Annealing time (minutes) **  
    - 简介：退火时间  
    - 数据类型：字符串  
    - 缺失/空值："na"表示缺失  
    - 取值语义：5-120分钟范围值或" "（空）  
    - 依赖/互斥：与"Annealing ambience"关联  
    - 允许操作：分箱复用  

11. **Annealing ambience**  
    - 简介：退火环境  
    - 数据类型：字符串  
    - 缺失/空值："na"表示缺失  
    - 取值语义："glovebox"（手套箱）或"na"  
    - 依赖/互斥：与"Annealing time"关联  
    - 允许操作：枚举值复用  

12. **HTL**  
    - 简介：空穴传输层材料  
    - 数据类型：字符串  
    - 缺失/空值：无  
    - 取值语义："spiro OMeTAD"/"P3HT"/"no"  
    - 依赖/互斥：与"HTL additive"关联  
    - 允许操作：枚举值复用  

13. **HTL additive **  
    - 简介：空穴传输层添加剂  
    - 数据类型：字符串  
    - 缺失/空值：无  
    - 取值语义："Li+TBP"/"Li+D TBP"/"no"  
    - 依赖/互斥：与"HTL"关联  
    - 允许操作：枚举值复用  

14. **PCEstabilized (%)**  
    - 简介：稳定化光电转换效率  
    - 数据类型：字符串  
    - 缺失/空值：无  
    - 取值语义：10-25%范围值  
    - 依赖/互斥：无  
    - 允许操作：分箱复用  

**no vs 0 说明**  
- "no"仅出现在：Anti solvent treatment、HTL、HTL additive  
- "0"仅出现在：ETL 2（表示未应用第二层）  
- 数值列（如温度、时间）的"0"视为无效值，需替换为"na"