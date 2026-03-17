"""
Generate a dataset of ~250 Chinese occupations using an LLM via OpenRouter.

Generates occupations industry-by-industry with structured data including
job titles, salaries, employment numbers, and education requirements.
Results are cached incrementally to occupations.json so the script can
be resumed if interrupted.

Usage:
    uv run python cn/generate_occupations.py
    uv run python cn/generate_occupations.py --model google/gemini-3-flash-preview
"""

import argparse
import json
import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "google/gemini-3-flash-preview"
OUTPUT_FILE = "cn/occupations.json"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Categories with target occupation count. Counts are weighted to roughly
# reflect each sector's share of China's 734 million employed (2024 data).
# Agriculture ~22.8%, Industry ~29.1%, Services ~48.1% of employment.
CATEGORIES = [
    ("制造业", 22),        # largest industrial employer, undergoing "machine-for-human" transformation
    ("信息技术", 18),      # includes digital economy, AI, platform tech
    ("医疗卫生", 18),      # aging population driving healthcare expansion
    ("教育", 16),          # large public sector employer
    ("金融", 15),          # banking, insurance, securities
    ("建筑工程", 18),      # major industrial employer, includes real estate services
    ("交通运输", 16),      # logistics, delivery, ride-hailing, new energy vehicles
    ("农林牧渔", 16),      # still ~22.8% of employment but declining
    ("商业服务", 16),      # consulting, legal services, accounting firms, HR
    ("餐饮住宿", 15),      # food service, hotels, tourism
    ("文化传媒", 15),      # includes new media, gaming, live-streaming
    ("法律与公共管理", 14), # government, public safety, legal profession
    ("销售零售与电商", 17), # includes cross-border e-commerce, live commerce
    ("个人与生活服务", 16), # gig economy, pet services, eldercare, platform workers
    ("新能源与环保", 12),   # green jobs: solar, EV, storage, carbon management
]

SYSTEM_PROMPT = """\
You are generating a realistic dataset of Chinese occupations for data visualization. \
You will be given a Chinese industry category and asked to produce a list of occupations \
in that industry.

Use the following key reference data from China's 2024 national statistics and the 2022 \
Occupational Classification Dictionary (职业分类大典) to ensure realism:

MACRO EMPLOYMENT (2024):
- Total employed: 734.39 million (down 6.02 million from 2023)
- Urban employed: 473.45 million (64.5% of total)
- Migrant workers: 299.73 million
- Flexible/gig workers: over 200 million
- Labor productivity: ¥173,898/person/year
- GDP breakdown: agriculture 6.8%, industry 36.5%, services 56.7%
- Employment breakdown: agriculture ~22.8%, industry ~29.1%, services ~48.1%

SALARY BENCHMARKS (annual, RMB):
- Migrant workers (location-based gig): ~96,000-180,000 (monthly 8k-15k)
- Cloud-based gig workers (high end): 180,000+ (monthly 15k+)
- Factory workers: 50,000-90,000
- Tech/IT professionals: 120,000-400,000+
- Medical professionals: 80,000-300,000+
- Teachers: 70,000-150,000
- Senior managers: 200,000-600,000+
- Agricultural workers: 20,000-50,000
- National average labor productivity: ~174,000/person/year

EMERGING SECTORS (from 2024-2025 new occupation announcements):
- AI: 生成式人工智能系统应用员, cloud-native ops
- New energy: 储能电站运维管理员, hydrogen reduction, EV testing
- Low-altitude economy: drone fleet planning
- Health: sleep health management, online doctors (+50% demand)
- Gig economy: 陪诊师 (+30%), 宠物伴宠师 (+70%), 跨境电商运营管理师
- Digital economy: 97 officially designated "digital occupations", 133 "green occupations"

For each occupation, provide:
- title: Chinese occupation name (e.g. "软件工程师")
- slug: pinyin slug using hyphens, no tones (e.g. "ruanjian-gongchengshi")
- category: the industry category (exactly as given)
- median_pay_annual: annual median salary in RMB. Use realistic figures calibrated to \
the benchmarks above. Low-skill jobs can be 25,000-60,000; mid-skill 60,000-150,000; \
high-skill 150,000-500,000+.
- num_jobs: estimated number of people employed in this occupation across all of China. \
The sum across ALL ~250 occupations should roughly account for the 734 million total. \
Large categories like "农民" or "建筑工人" can have tens of millions; niche roles may \
have tens of thousands.
- education: one of exactly: "初中及以下", "高中/中专", "大专", "本科", "硕士", "博士"
- outlook_pct: projected annual growth rate as integer percentage (e.g. 5 means 5%). \
Use negative values for declining occupations. Align with macro trends: agriculture \
declining, traditional manufacturing flat/declining, AI/new energy/healthcare growing fast.
- outlook_desc: one of exactly: "下降", "基本不变", "低于平均", "平均水平", "高于平均", "远高于平均"

IMPORTANT GUIDELINES:
- Include both traditional occupations AND newly emerged ones (e.g. 网约车司机, 外卖骑手, \
直播带货主播, AI训练师, 数据标注员).
- For agriculture, include large-employment traditional roles (种植业农民, 畜牧养殖员) \
with realistically large num_jobs (tens of millions each).
- For manufacturing, reflect the "machine-for-human" transition: include both traditional \
assembly workers (large employment, declining outlook) and advanced manufacturing roles \
(smaller employment, growing).
- Salary should reflect China's reality, NOT US/Western levels.

Respond with ONLY a JSON array of objects, no other text. Do not wrap in markdown code fences.\
"""


# Per-category guidance to help the LLM generate more realistic data
CATEGORY_HINTS = {
    "制造业": "中国制造业正经历'机器换人'转型。包括传统流水线工人（就业人数大，前景下降）、"
              "高技术制造如装备制造/芯片/新能源汽车（增长快）、质检员、焊工、数控机床操作员、"
              "工业机器人运维员等。规模以上装备制造业和高技术制造业比重不断攀升。",
    "信息技术": "数字经济规模超60万亿元。包括软件开发、AI工程师、数据标注员（七大数据标注基地）、"
                "云计算运维、网络安全、大数据分析、生成式AI应用员、区块链等。97个官方认定数字职业。"
                "算力规模280 EFLOPS。",
    "医疗卫生": "老龄化加速（60岁以上人口3.1亿，占22%）驱动医疗需求。包括医生、护士、药剂师、"
                "中医、互联网医生（需求增速近50%）、睡眠健康管理师、康复治疗师、医疗器械技术员等。",
    "教育": "包括中小学教师、大学教授、职业培训师、早教教师、在线教育讲师、特殊教育教师等。"
            "职业教育体系正向数字化和绿色化转型。",
    "金融": "包括银行柜员（数量下降）、理财顾问、证券分析师、保险代理人、风控专员、"
            "金融科技工程师、量化交易员等。基金/证券从业人员。",
    "建筑工程": "建筑业是重要工业雇主。包括建筑工人（数千万）、土木工程师、建筑设计师、"
               "监理工程师、装修工人、BIM工程师、工程造价师等。房地产转型中。",
    "交通运输": "包括货车司机、网约车司机（灵活就业代表）、外卖骑手（近千万）、快递员、"
               "公交/地铁司机、飞行员、船员、物流管理、智能网联汽车测试员等。"
               "近3亿农民工中很多从事交通运输。",
    "农林牧渔": "占就业22.8%但仅创造6.8%GDP，存在大量隐性失业。包括种植业农民（数千万）、"
               "畜牧养殖员、渔民、林业工人、农机操作员、农业技术员、兽医等。"
               "劳动力持续向城镇转移。每个职业的num_jobs要大（农民可达上亿）。",
    "商业服务": "包括会计师、审计师、人力资源管理、企业管理咨询、广告营销、翻译、"
               "知识产权代理、商务秘书等现代生产性服务业。",
    "餐饮住宿": "就业'海绵'行业。包括厨师、服务员、酒店管理、调酒师、面点师、"
               "外卖商家运营、连锁餐饮店长等。大量灵活就业。",
    "文化传媒": "包括记者、编辑、网络主播（已成为正式新职业）、短视频创作者、游戏设计师、"
               "影视制作、动画设计师、文创产品策划运营师、版权经纪人等。"
               "直播带货和内容创作是高增长领域。",
    "法律与公共管理": "包括公务员、警察、法官、律师、公证员、消防员、社区工作者、"
                    "城市管理执法员等。政府公共服务数字化转型中。",
    "销售零售与电商": "包括传统零售店员、电商运营、直播带货主播、跨境电商运营管理师、"
                    "用户增长运营师、商品采购员、市场营销员等。"
                    "平台经济驱动新型销售岗位快速增长。",
    "个人与生活服务": "灵活就业超2亿人的核心领域。包括家政服务员、月嫂/育婴师、"
                    "养老护理员、美容美发师、健身教练、陪诊师（增长30%）、"
                    "宠物伴宠师（增长70%）、保安员、物业管理员等。"
                    "位置型灵活就业月薪8k-15k。",
    "新能源与环保": "133个官方认定绿色职业。包括光伏电站运维员、风电工程师、"
                  "储能电站运维管理员、碳资产管理师、新能源汽车维修技师、"
                  "氢能技术员、环境监测工程师、垃圾分类管理员等。"
                  "工业互联网产业规模超1.5万亿。新能源是远高于平均增长领域。",
}


def generate_batch(client, category, count, model):
    """Generate occupations for one industry category."""
    hint = CATEGORY_HINTS.get(category, "")
    prompt = (
        f"请为\u201c{category}\u201d行业生成 {count} 个有代表性的中国职业。"
        f"确保涵盖该行业从基层到高层的各类职业。\n\n"
        f"该行业的具体参考信息：{hint}"
    )
    response = client.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        },
        timeout=120,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    # Strip markdown code fences if present
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    return json.loads(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if category already cached")
    args = parser.parse_args()

    # Load existing occupations grouped by category
    existing = {}
    if os.path.exists(OUTPUT_FILE) and not args.force:
        with open(OUTPUT_FILE) as f:
            for occ in json.load(f):
                cat = occ.get("category", "")
                if cat not in existing:
                    existing[cat] = []
                existing[cat].append(occ)

    all_occupations = []
    # Add already-cached occupations
    for cat_occs in existing.values():
        all_occupations.extend(cat_occs)

    print(f"Generating occupations with {args.model}")
    print(f"Already cached categories: {list(existing.keys())}")

    errors = []
    client = httpx.Client()

    for i, (category, count) in enumerate(CATEGORIES):
        if category in existing:
            print(f"  [{i+1}/{len(CATEGORIES)}] SKIP {category} (cached, {len(existing[category])} occupations)")
            continue

        print(f"  [{i+1}/{len(CATEGORIES)}] {category} ({count} occupations)...", end=" ", flush=True)

        try:
            batch = generate_batch(client, category, count, args.model)
            # Ensure category is set correctly
            for occ in batch:
                occ["category"] = category
            all_occupations.extend(batch)
            print(f"got {len(batch)}")
        except Exception as e:
            print(f"ERROR: {e}")
            errors.append(category)

        # Save after each batch (incremental checkpoint)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_occupations, f, ensure_ascii=False, indent=2)

        if i < len(CATEGORIES) - 1:
            time.sleep(args.delay)

    client.close()

    # Deduplicate by slug
    seen = {}
    unique = []
    for occ in all_occupations:
        if occ["slug"] not in seen:
            seen[occ["slug"]] = True
            unique.append(occ)
    all_occupations = unique

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_occupations, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Generated {len(all_occupations)} occupations, {len(errors)} errors.")
    if errors:
        print(f"Errors: {errors}")

    # Summary
    by_cat = {}
    for occ in all_occupations:
        cat = occ["category"]
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print("\nBy category:")
    for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
