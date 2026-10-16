# 日最高温模型（香港 / 巴黎 / 纽约）

写在本机 `/Users/jasperZ@bixin.com/zxmoney/temp-ges`，不属于 interplanetbit 仓库。

| 城市 | 结算站 | 单位 | 实况 |
|------|--------|------|------|
| 香港 | HKO 总部 Absolute Daily Max | °C，一位小数 | 天文台区域读数 |
| 巴黎 | Le Bourget `LFPB` | °C，整数 | METAR + Open-Meteo 预报 |
| 纽约 | LaGuardia `KLGA` | °F，整数 | METAR + Open-Meteo 预报 |

香港：30.2°C 落在 30°C 档，31.9°C 仍在 31°C 档，32.0°C 才进入 32°C 档。默认盘口 24°C 或以下 … 34°C 或以上。赤鱲角不是结算站。

巴黎 / 纽约：Polymarket 按整度结算；盘口默认巴黎 14–28°C、纽约 55–85°F。

## 运行

```bash
cd /Users/jasperZ@bixin.com/zxmoney/temp-ges
python3 -m hk_temp_model
python3 -m hk_temp_model --cities hk,paris,nyc
python3 -m unittest discover -s hk_temp_model/tests -t .
```

香港日内用总部「午夜至今最高」做下限；辐射、紫外和风取京士柏。巴黎/纽约用当日 METAR 最高做下限。

模型给出各档公允，不下单。

`--watch 10` 会一直跑。状态分别写入：

- `hk_temp_model/data/online_state.json`
- `hk_temp_model/data/online_state_paris.json`
- `hk_temp_model/data/online_state_nyc.json`

结算值出现后，用最接近当地 16:00 的预测对答案，微调偏差和波动。同一天只更新一次。
