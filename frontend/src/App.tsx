import React, { useState } from 'react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Legend, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import dashboardData from './data/dashboard-data.json';

// Conversation example interface
interface ConversationExample {
  id: string;
  category: string;
  mainQuestion: string;
  insights: string;
  subcategory: string;
  totalMessages: number;
}

// Color mappings (UI layer)
const CATEGORY_COLORS: Record<string, string> = {
  'GNSS': '#6366f1',
  'SOFTWARE': '#8b5cf6',
  'OTRO': '#a78bfa',
  'ÓPTICA': '#c4b5fd',
  'OPTICA': '#c4b5fd',
  'COLECTORA': '#0ea5e9',
  'UAS': '#22d3d1',
  'ESCÁNER': '#10b981',
  'ESCANER': '#10b981',
  'RADIO': '#34d399'
};

const RESOLUTION_COLORS: Record<string, string> = {
  'Resuelto': '#10b981',
  'Parcial': '#f59e0b',
  'Sin Seguimiento': '#6b7280',
  'No Resuelto': '#ef4444'
};

const SATISFACTION_COLORS: Record<string, string> = {
  'Alta': '#10b981',
  'Media': '#f59e0b',
  'No Determinable': '#6b7280',
  'Baja': '#ef4444'
};

// Extract and transform data from imported JSON
const { kpis, tendenciaMensual, tiposProblema, productos, radarData, conversationExamples } = dashboardData;

// Apply colors to data that needs them
const categorias = dashboardData.categorias.map(cat => ({
  ...cat,
  color: CATEGORY_COLORS[cat.name] || '#94a3b8'
}));

const resolucion = dashboardData.resolucion.map(res => ({
  ...res,
  color: RESOLUTION_COLORS[res.name] || '#94a3b8'
}));

const satisfaccion = dashboardData.satisfaccion.map(sat => ({
  ...sat,
  color: SATISFACTION_COLORS[sat.name] || '#94a3b8'
}));

const KPICard = ({ title, value, subtitle, gradient }) => (
  <div className={`relative overflow-hidden rounded-2xl p-6 ${gradient} shadow-xl`}>
    <div className="relative z-10">
      <p className="text-white/80 text-sm font-medium tracking-wide uppercase">{title}</p>
      <p className="text-white text-4xl font-bold mt-2">{value}</p>
      <p className="text-white/60 text-sm mt-1">{subtitle}</p>
    </div>
  </div>
);

const SectionTitle = ({ children }) => (
  <div className="flex items-center gap-3 mb-6">
    <h2 className="text-2xl font-bold text-slate-800">{children}</h2>
  </div>
);

interface ConversationCardProps {
  example: ConversationExample;
  currentIndex: number;
  total: number;
  onPrevious: () => void;
  onNext: () => void;
}

const ConversationCard: React.FC<ConversationCardProps> = ({
  example,
  currentIndex,
  total,
  onPrevious,
  onNext
}) => {
  const categoryColor = CATEGORY_COLORS[example.category] || '#94a3b8';

  return (
    <div className="bg-white rounded-3xl shadow-xl p-8 border border-slate-100 flex flex-col h-auto md:h-[520px]">
      {/* Header con badge de categoría */}
      <div className="flex items-center justify-between mb-6">
        <SectionTitle>Ejemplos de Conversaciones</SectionTitle>
        <div
          className="px-4 py-2 rounded-full text-white font-semibold text-sm shadow-md"
          style={{ backgroundColor: categoryColor }}
        >
          {example.category}
        </div>
      </div>

      {/* Pregunta principal */}
      <div className="mb-6">
        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-2">
          Consulta Principal
        </h3>
        <p className="text-xl font-bold text-slate-800 leading-tight">
          {example.mainQuestion}
        </p>
        <p className="text-sm text-slate-500 mt-2">
          {example.subcategory} · {example.totalMessages} mensajes
        </p>
      </div>

      {/* Insights */}
      <div className="mb-8 flex-grow overflow-y-auto pr-2 custom-scrollbar">
        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-2">
          Resumen del Caso
        </h3>
        <p className="text-slate-700 leading-relaxed">
          {example.insights}
        </p>
      </div>

      {/* Controles de navegación */}
      <div className="flex items-center justify-between pt-6 border-t border-slate-200">
        <button
          onClick={onPrevious}
          className="flex items-center gap-2 px-5 py-3 rounded-xl bg-gradient-to-r from-slate-100 to-slate-200 hover:from-slate-200 hover:to-slate-300 text-slate-700 font-medium transition-all duration-200 shadow-sm hover:shadow-md"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Anterior
        </button>

        <div className="flex items-center gap-2">
          <span className="text-slate-600 font-semibold">
            {currentIndex + 1} / {total}
          </span>
        </div>

        <button
          onClick={onNext}
          className="flex items-center gap-2 px-5 py-3 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-600 hover:from-indigo-600 hover:to-violet-700 text-white font-medium transition-all duration-200 shadow-md hover:shadow-lg"
        >
          Siguiente
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default function ReporteSoporte2025() {
  const [activeTab, setActiveTab] = useState('overview');
  const [currentExampleIndex, setCurrentExampleIndex] = useState(0);

  // Carousel navigation handlers
  const handlePrevExample = () => {
    setCurrentExampleIndex((prev) =>
      prev === 0 ? conversationExamples.length - 1 : prev - 1
    );
  };

  const handleNextExample = () => {
    setCurrentExampleIndex((prev) =>
      prev === conversationExamples.length - 1 ? 0 : prev + 1
    );
  };

  const currentExample = conversationExamples[currentExampleIndex];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-slate-100 to-indigo-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-slate-900 via-indigo-950 to-violet-950 text-white">
        <div className="max-w-7xl mx-auto px-6 py-10">
          <div className="flex items-center gap-4 mb-2">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">Reporte de Soporte</h1>
              <p className="text-indigo-300 text-lg">GEOCOM · WhatsApp Soporte GNSS-Óptica · 2025</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-10">
        {/* KPIs */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <KPICard
            title="Conversaciones"
            value="559"
            subtitle="clientes atendidos"
            gradient="bg-gradient-to-br from-gray-500 to-slate-700"
          />
          <KPICard
            title="Mensajes Totales"
            value="32,136"
            subtitle="intercambiados"
            gradient="bg-gradient-to-br from-gray-500 to-slate-700"
          />
          <KPICard
            title="Promedio/Chat"
            value="57.5"
            subtitle="mensajes por caso"
            gradient="bg-gradient-to-br from-gray-500 to-slate-700"
          />
          <KPICard
            title="Satisfacción Alta"
            value="47%"
            subtitle="264 clientes"
            gradient="bg-gradient-to-br from-gray-500 to-slate-700"
          />
        </div>

        {/* Charts Row 1 */}
        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          {/* Categorías */}
          <div className="bg-white rounded-3xl shadow-xl p-8 border border-slate-100">
            <SectionTitle>Casos por Categoría</SectionTitle>
            <ResponsiveContainer width="100%" height={320}>
              <PieChart>
                <Pie
                  data={categorias}
                  cx="50%"
                  cy="50%"
                  innerRadius={70}
                  outerRadius={120}
                  paddingAngle={3}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {categorias.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div className="mt-4 grid grid-cols-4 gap-2">
              {categorias.map((cat, i) => (
                <div key={i} className="flex items-center gap-2 text-sm">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: cat.color }}></div>
                  <span className="text-slate-600">{cat.name}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Resolución */}
          <div className="bg-white rounded-3xl shadow-xl p-8 border border-slate-100">
            <SectionTitle>Tasa de Resolución</SectionTitle>
            <div className="grid grid-cols-2 gap-4 mt-4">
              {resolucion.map((item, i) => (
                <div key={i} className="text-center p-5 rounded-2xl bg-slate-50">
                  <div 
                    className="w-16 h-16 mx-auto rounded-full flex items-center justify-center text-white text-xl font-bold mb-3 shadow-lg"
                    style={{ backgroundColor: item.color }}
                  >
                    {item.pct}
                  </div>
                  <p className="font-semibold text-slate-800">{item.name}</p>
                  <p className="text-slate-500 text-sm">{item.value} casos</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Tendencia Mensual */}
        <div className="bg-white rounded-3xl shadow-xl p-8 border border-slate-100 mb-12">
          <SectionTitle>Tendencia Mensual 2025</SectionTitle>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={tendenciaMensual}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="mes" stroke="#64748b" />
              <YAxis yAxisId="left" stroke="#6366f1" />
              <YAxis yAxisId="right" orientation="right" stroke="#10b981" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: 'none', 
                  borderRadius: '12px',
                  color: 'white'
                }}
              />
              <Legend />
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="conversaciones" 
                stroke="#6366f1" 
                strokeWidth={3}
                dot={{ fill: '#6366f1', strokeWidth: 2, r: 6 }}
                name="Conversaciones"
              />
              <Line 
                yAxisId="right"
                type="monotone" 
                dataKey="mensajes" 
                stroke="#10b981" 
                strokeWidth={3}
                dot={{ fill: '#10b981', strokeWidth: 2, r: 6 }}
                name="Mensajes"
              />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-4 p-4 bg-amber-50 rounded-xl border border-amber-200">
            <p className="text-amber-800 text-sm">
              Enero fue el mes con mayor actividad (96 conversaciones). 
              Se observa una tendencia estable durante el año con leve incremento en junio.
            </p>
          </div>
        </div>

        {/* Charts Row 2 */}
        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          {/* Tipos de Problema */}
          <div className="bg-white rounded-3xl shadow-xl p-8 border border-slate-100">
            <SectionTitle>Tipos de Problema</SectionTitle>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={tiposProblema} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" stroke="#64748b" />
                <YAxis dataKey="tipo" type="category" width={110} stroke="#64748b" tick={{ fontSize: 12 }} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1e293b', 
                    border: 'none', 
                    borderRadius: '12px',
                    color: 'white'
                  }}
                />
                <Bar dataKey="cantidad" fill="url(#gradientBar)" radius={[0, 8, 8, 0]} />
                <defs>
                  <linearGradient id="gradientBar" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#6366f1" />
                    <stop offset="100%" stopColor="#8b5cf6" />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Productos más consultados */}
          <div className="bg-white rounded-3xl shadow-xl p-8 border border-slate-100">
            <SectionTitle>Productos Más Consultados</SectionTitle>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={productos}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="producto" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1e293b', 
                    border: 'none', 
                    borderRadius: '12px',
                    color: 'white'
                  }}
                />
                <Bar dataKey="consultas" fill="url(#gradientBar2)" radius={[8, 8, 0, 0]} />
                <defs>
                  <linearGradient id="gradientBar2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#0ea5e9" />
                    <stop offset="100%" stopColor="#6366f1" />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Satisfacción + Radar */}
        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          {/* Satisfacción */}
          <div className="bg-white rounded-3xl shadow-xl p-8 border border-slate-100">
            <SectionTitle>Nivel de Satisfacción</SectionTitle>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={satisfaccion}
                  cx="50%"
                  cy="50%"
                  outerRadius={110}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {satisfaccion.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div className="mt-4 flex justify-center gap-4">
              {satisfaccion.map((sat, i) => (
                <div key={i} className="flex items-center gap-2 text-sm">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: sat.color }}></div>
                  <span className="text-slate-600">{sat.name}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Radar de desempeño */}
          <div className="bg-white rounded-3xl shadow-xl p-8 border border-slate-100">
            <SectionTitle>Indicadores de Desempeño</SectionTitle>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#e2e8f0" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 12 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} />
                <Radar
                  name="Desempeño"
                  dataKey="A"
                  stroke="#6366f1"
                  fill="#6366f1"
                  fillOpacity={0.5}
                />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Insights Finales */}
        <div className="bg-gradient-to-br from-slate-900 via-indigo-950 to-violet-950 rounded-3xl shadow-xl p-8 text-white">
          <SectionTitle><span className="text-white">Consideraciones</span></SectionTitle>
          <div className="grid md:grid-cols-3 gap-6 mt-6">
            <div className="bg-white/10 backdrop-blur rounded-2xl p-6">
              <h3 className="font-bold text-lg mb-2">Tasa de Resolución</h3>
              <p className="text-slate-300 text-sm">
                74% de los casos fueron resueltos total o parcialmente. Solo 5% quedó sin resolver.
                Excelente ratio para soporte técnico especializado.
              </p>
            </div>
            <div className="bg-white/10 backdrop-blur rounded-2xl p-6">
              <h3 className="font-bold text-lg mb-2">Área de Mejora</h3>
              <p className="text-slate-300 text-sm">
                20% de conversaciones sin seguimiento. Implementar sistema de cierre
                obligatorio y encuestas de satisfacción post-atención.
              </p>
            </div>
            <div className="bg-white/10 backdrop-blur rounded-2xl p-6">
              <h3 className="font-bold text-lg mb-2">Foco 2026</h3>
              <p className="text-slate-300 text-sm">
                TBC y equipos GNSS (R12i, R8) concentran la mayoría de consultas.
                Crear base de conocimiento y videos tutoriales para estos productos.
              </p>
            </div>
          </div>
        </div>

        {/* Ejemplos de Conversaciones */}
        {conversationExamples && conversationExamples.length > 0 && (
          <div className="mb-12 mt-12">
            <ConversationCard
              example={currentExample}
              currentIndex={currentExampleIndex}
              total={conversationExamples.length}
              onPrevious={handlePrevExample}
              onNext={handleNextExample}
            />
          </div>
        )}

        {/* Footer */}
        <footer className="mt-12 text-center text-slate-500 text-sm">
          <p>Reporte generado automáticamente · Análisis de 559 conversaciones · Período: Enero - Diciembre 2025</p>
          <p className="mt-1">GEOCOM ·Soporte GNSS-Óptica 2025</p>
        </footer>
      </main>
    </div>
  );
}
