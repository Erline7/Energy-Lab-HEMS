<template>
  <div class="dataset-upload-card">
    <div class="control-group">
      <label>{{ store.isZh ? '📂 导入能源数据集' : '📂 Import Dataset' }}</label>
      <select v-model="selectedType" class="styled-select">
        <option value="electricity">{{ store.isZh ? '⚡ 电力与环境主数据' : '⚡ Electricity & Weather' }}</option>
        <option value="water_main">{{ store.isZh ? '💧 全屋主水表数据' : '💧 Main Water Meter' }}</option>
        <option value="water_sub">{{ store.isZh ? '🚰 用水设备(子水表)' : '🚰 Sub Water Meter' }}</option>
        <option value="gas">{{ store.isZh ? '🔥 燃气月度消耗数据' : '🔥 Natural Gas Data' }}</option>
      </select>
    </div>

    <div class="control-group" style="margin-top: 15px;">
      <button @click="$refs.csvInputRef.click()" class="action-btn" :disabled="isUploading">
        <span v-if="!isUploading">
          📤 {{ store.isZh ? '选择并上传 CSV 文件' : 'Select & Upload CSV' }}
        </span>
        <span v-else>
          ⏳ {{ store.isZh ? '正在清洗并入库...' : 'Processing...' }}
        </span>
      </button>
      <input type="file" ref="csvInputRef" @change="handleDatasetUpload" accept=".csv" style="display: none" />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import api from '../utils/api.js';
import { useGlobalStore } from '../store/global.js';

const store = useGlobalStore();
const emit = defineEmits(['upload-success']);

const selectedType = ref('electricity');
const csvInputRef = ref(null);
const isUploading = ref(false);

const handleDatasetUpload = async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  isUploading.value = true;
  const formData = new FormData();
  formData.append('dataset_type', selectedType.value);
  formData.append('file', file);

  try {
    const response = await api.post('/api/upload_dataset', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });

    // 触发成功事件，通知 Chat.vue
    emit('upload-success');

  } catch (error) {
    alert(store.isZh ? "上传失败：" : "Upload failed: " + (error.response?.data?.detail || error.message));
  } finally {
    isUploading.value = false;
    e.target.value = ''; // 清空输入框，方便下次上传同名文件
  }
};
</script>

<style scoped>
.dataset-upload-card {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px dashed #ccc;
}
.control-group label {
  display: block;
  font-size: 14px;
  font-weight: bold;
  margin-bottom: 8px;
  color: #2c3e50;
}
.styled-select {
  width: 100%;
  padding: 8px;
  border-radius: 6px;
  border: 1px solid #ddd;
  background-color: #fff;
}
.action-btn {
  width: 100%;
  padding: 10px;
  background: #ecf0f1;
  color: #2c3e50;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: bold;
  transition: background-color 0.2s;
}
.action-btn:hover:not(:disabled) {
  background: #d1d8dd;
}
.action-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}
</style>