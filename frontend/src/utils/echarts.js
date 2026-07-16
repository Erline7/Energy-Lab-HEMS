// src/utils/echarts.js

// 强力家电名称中英转换器
export const translateApp = (name, isZh) => {
  // 翻译为英文的字典（涵盖代码、中文、英文）
  const toEn = {
    'HPE': 'Heat Pump', '热泵': 'Heat Pump', 'Heat Pump': 'Heat Pump',
    'CDE': 'Dryer',     '干衣机': 'Dryer',     'Dryer': 'Dryer',
    'CWE': 'Washer',    '洗衣机': 'Washer',    'Washer': 'Washer',
    'FGE': 'Fridge',    '冰箱': 'Fridge',      'Fridge': 'Fridge',
    'WOE': 'Oven',      '烤箱': 'Oven',        'Oven': 'Oven',
    'TVE': 'TV',        '电视': 'TV',          'TV': 'TV',
    'BME': 'Lighting',  '照明': 'Lighting',    'Lighting': 'Lighting'
  };

  // 翻译为中文的字典
  const toZh = {
    'HPE': '热泵',   '热泵': '热泵',   'Heat Pump': '热泵',
    'CDE': '干衣机', '干衣机': '干衣机', 'Dryer': '干衣机',
    'CWE': '洗衣机', '洗衣机': '洗衣机', 'Washer': '洗衣机',
    'FGE': '冰箱',   '冰箱': '冰箱',   'Fridge': '冰箱',
    'WOE': '烤箱',   '烤箱': '烤箱',   'Oven': '烤箱',
    'TVE': '电视',   '电视': '电视',   'TV': '电视',
    'BME': '照明',   '照明': '照明',   'Lighting': '照明'
  };

  return isZh ? (toZh[name] || name) : (toEn[name] || name);
};

// 下面的 generateTimeLabels 函数保持不变...
export const generateTimeLabels = () => {
  const labels = [];
  for (let i = 0; i < 96; i++) {
    labels.push(`${Math.floor(i / 4).toString().padStart(2, '0')}:${((i % 4) * 15).toString().padStart(2, '0')}`);
  }
  return labels;
};