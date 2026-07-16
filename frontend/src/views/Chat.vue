<template>
  <div class="chat-view">
    <aside class="chat-sidebar card">
      <div class="control-group">
        <label>{{ store.isZh ? '智能体引擎 (LLM)' : 'LLM Engine' }}</label>
        <select v-model="store.selectedModel" class="styled-select">
          <option value="qwen3.6-27b">Qwen3.6</option>
          <option value="kimi-k2.6">Kimi (2.6)</option>
          <option value="qwen3.7-max-2026-05-20">Qwen3.7 (Max)</option>
        </select>
      </div>

      <div class="control-group" style="margin-top: 20px;">
        <label>{{ store.isZh ? '知识库管理' : 'Knowledge Base' }}</label>
        <button @click="$refs.fileInputRef.click()" class="action-btn">
          🔗 {{ store.isZh ? '上传 PDF 说明书' : 'Upload PDF' }}
        </button>
        <input type="file" ref="fileInputRef" @change="handleFileUpload" accept=".pdf" style="display: none" />
      </div>

      <div style="margin-top: 20px;">
        <DatasetUpload @upload-success="onDatasetUploaded" />
      </div>
    </aside>

    <main class="chat-main card">
      <div class="chat-header">🤖 {{ store.isZh ? '能源管家对话' : 'Energy Butler Chat' }}</div>

      <div class="message-list" ref="chatBox">
        <div v-for="(msg, i) in messages" :key="i" :class="['msg', msg.role]">
          <div class="avatar">{{ msg.role === 'user' ? 'U' : 'AI' }}</div>
          <div class="bubble" v-html="formatText(msg.text)"></div>
        </div>
        <div v-if="loading" class="msg ai typing-indicator">
          <div class="avatar">AI</div>
          <div class="bubble">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
          </div>
        </div>
      </div>

      <div v-if="pdfProgress.show" class="progress-bar">
        <span>{{ pdfProgress.text }}</span>
        <div class="track"><div class="fill" :style="{ width: pdfProgress.percent + '%' }"></div></div>
      </div>

      <div class="input-area">
        <textarea
          v-model="userInput"
          @keydown.enter.prevent="handleSend"
          :placeholder="store.isZh ? '询问用电建议或说明书内容...' : 'Ask about energy or manuals...'"
        ></textarea>
        <button @click="handleSend" :disabled="loading" class="send-btn">
          {{ store.isZh ? '发送' : 'Send' }}
        </button>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, nextTick, watch } from 'vue';
import api from '../utils/api.js';
import { useGlobalStore } from '../store/global.js';
import DatasetUpload from '../components/DatasetUpload.vue';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
const store = useGlobalStore();

const fileInputRef = ref(null);
const chatBox = ref(null);
const userInput = ref('');
const loading = ref(false);
const pdfProgress = ref({ show: false, percent: 0, text: '' });
let progressInterval = null;

const zhWelcome = '您好！我是 AI 能源管家，您可以随时提问。';
const enWelcome = 'Hello! I am your AI Butler.';

// 初始化消息列表
const messages = ref([
  { role: 'ai', text: store.isZh ? zhWelcome : enWelcome }
]);

watch(() => store.isZh, (newIsZh) => {
  if (messages.value.length > 0) {
    const firstMsgText = messages.value[0].text;
    if (firstMsgText === zhWelcome || firstMsgText === enWelcome) {
      messages.value[0].text = newIsZh ? zhWelcome : enWelcome;
    }
  }
});

const formatText = (text) => {
  if (!text) return '';
  const rawHtml = marked.parse(text, {
    breaks: true,
    gfm: true
  });
  return DOMPurify.sanitize(rawHtml);
};

// 👇 处理数据集上传成功的回调逻辑
const onDatasetUploaded = async () => {
  const successMsg = store.isZh
    ? '您的能源数据集已成功清洗并导入数据库！我的大模型上下文已经同步，您可以直接向我提问关于这批新数据的分析情况了。'
    : 'Dataset successfully cleaned and imported! My context is synced, you can now ask me to analyze this new data.';

  // AI 主动发送一条消息反馈给用户
  messages.value.push({ role: 'ai', text: successMsg });
  await nextTick(() => chatBox.value.scrollTop = chatBox.value.scrollHeight);

  // 如果你有全局刷新的需求，可以取消注释下面的代码
  // store.dispatch('refreshDashboardData');
};

const handleSend = async () => {
  if (!userInput.value.trim() || loading.value) return;
  const content = userInput.value;
  messages.value.push({ role: 'user', text: content });
  const payloadContent = store.isZh
    ? content
    : content + '\n\n(System strict instruction: Please provide your entire response in English.)';

  userInput.value = '';
  loading.value = true;
  await nextTick(() => chatBox.value.scrollTop = chatBox.value.scrollHeight);

  try {
    const res = await api.post('/api/chat', {
      message: payloadContent,
      selected_date: store.selectedDate || null, // 确保空值安全
      model: store.selectedModel,
      session_id: store.sessionId || "default"   // 🌟 补全 session_id 关联
    });
    messages.value.push({ role: 'ai', text: res.data.reply });

    const replyStr = res.data.reply || '';
    if (
      replyStr.includes("视觉识别") ||
      replyStr.includes("后台排队") ||
      replyStr.includes("识别")
    ) {
      startPollingProgress();
    }
  } catch (err) {
    messages.value.push({ role: 'ai', text: store.isZh ? '请求失败，请检查后端。' : 'Error connecting to backend.' });
  } finally {
    loading.value = false;
    await nextTick(() => chatBox.value.scrollTop = chatBox.value.scrollHeight);
  }
};

const handleFileUpload = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append('file', file);

  messages.value.push({ role: 'ai', text: `正在接收文件: ${file.name}...` });
  try {
    const res = await api.post('/api/upload', formData);
    userInput.value = `我上传了[具体名称]说明书：${res.data.file_path}，请帮我读取。`;
  } catch (err) {
    messages.value.push({ role: 'ai', text: '文件上传失败。' });
  }
  e.target.value = '';
};

const startPollingProgress = () => {
  if (progressInterval) clearInterval(progressInterval);
  pdfProgress.value = { show: true, percent: 0, text: store.isZh ? '准备阅读...' : 'Preparing...' };

  progressInterval = setInterval(async () => {
    try {
      const res = await api.get('/api/progress');
      if (res.data.status === 'completed') {
        pdfProgress.value.percent = 100;
        pdfProgress.value.text = store.isZh ? '阅读完成！' : 'Completed!';
        clearInterval(progressInterval);
        setTimeout(() => pdfProgress.value.show = false, 3000);
      } else if (res.data.status === 'processing') {
        pdfProgress.value.percent = Math.round((res.data.current / res.data.total) * 100);
        pdfProgress.value.text = `Reading ${res.data.filename}...`;
      }
    } catch (e) {}
  }, 1000);
};
</script>

<style scoped>
.chat-view { display: flex; gap: 20px; padding: 20px; height: 100%; }
.card { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }

.chat-sidebar { width: 260px; padding: 20px; }
.control-group label { display: block; font-size: 14px; font-weight: bold; margin-bottom: 8px; color: #2c3e50; }
.styled-select { width: 100%; padding: 8px; border-radius: 6px; border: 1px solid #ddd; }
.action-btn { width: 100%; padding: 10px; background: #ecf0f1; border: none; border-radius: 6px; cursor: pointer; transition: 0.2s; }
.action-btn:hover { background: #d1d8dd; }

.chat-main { flex: 1; display: flex; flex-direction: column; }
.chat-header { padding: 15px 20px; font-size: 16px; font-weight: bold; border-bottom: 1px solid #eee; }
.message-list { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
.msg { display: flex; gap: 10px; max-width: 80%; }
.msg.user { align-self: flex-end; flex-direction: row-reverse; }
.avatar { width: 35px; height: 35px; background: #ecf0f1; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: bold;}
.user .avatar { background: #3498db; color: white; }
.bubble { padding: 12px 16px; background: #f0f2f5; border-radius: 12px; line-height: 1.0; font-size: 14px;}
.user .bubble { background: #3498db; color: white; }

.progress-bar { padding: 10px 20px; background: #f8f9fa; font-size: 12px; }
.track { height: 6px; background: #eee; border-radius: 3px; margin-top: 5px; }
.fill { height: 100%; background: #2ecc71; border-radius: 3px; transition: width 0.3s; }

.input-area { display: flex; gap: 10px; padding: 20px; border-top: 1px solid #eee; }
.input-area textarea { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 8px; resize: none; height: 45px; }
.send-btn { background: #27ae60; color: white; border: none; padding: 0 20px; border-radius: 8px; cursor: pointer; font-weight: bold; }
.typing-indicator .bubble {
  display: flex;
  gap: 4px;
  align-items: center;
  height: 44px;
  padding: 0 16px;
}

.dot {
  width: 6px;
  height: 6px;
  background-color: #94a3b8;
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out both;
}

.dot:nth-child(1) { animation-delay: -0.32s; }
.dot:nth-child(2) { animation-delay: -0.16s; }
.bubble :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 10px 0;
  background: white;
  color: #333;
  border-radius: 4px;
  overflow: hidden;
}

.bubble :deep(th), .bubble :deep(td) {
  border: 1px solid #ddd;
  padding: 8px 12px;
  text-align: left;
}

.bubble :deep(th) {
  background-color: #f8f9fa;
  font-weight: bold;
}

.user .bubble :deep(table) {
  color: #fff;
  background: rgba(255,255,255,0.1);
}
.user .bubble :deep(th), .user .bubble :deep(td) {
  border: 1px solid rgba(255,255,255,0.3);
}
@keyframes bounce {
  0%, 80%, 100% {
    transform: scale(0);
  }
  40% {
    transform: scale(1);
  }
}
</style>