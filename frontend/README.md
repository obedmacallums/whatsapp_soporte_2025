# Dashboard de Reportes WhatsApp

Dashboard interactivo construido con React, Vite, Recharts y Tailwind CSS para visualizar reportes de soporte técnico de WhatsApp Business.

## Características

- Dashboard responsivo con múltiples visualizaciones
- Gráficos interactivos usando Recharts (pie, bar, line, radar)
- Estilos modernos con Tailwind CSS
- Hot-reload automático con Vite
- TypeScript para type safety

## Requisitos

- Node.js v20 o superior
- npm 10 o superior

## Instalación

Las dependencias ya están instaladas, pero si necesitas reinstalarlas:

```bash
npm install
```

## Scripts Disponibles

### Desarrollo

Inicia el servidor de desarrollo con hot-reload:

```bash
npm run dev
```

La aplicación estará disponible en [http://localhost:5173/](http://localhost:5173/)

### Producción

Construye la aplicación optimizada para producción:

```bash
npm run build
```

Los archivos se generarán en el directorio `dist/`

### Vista Previa

Vista previa de la build de producción localmente:

```bash
npm run preview
```

## Estructura del Proyecto

```
frontend/
├── src/
│   ├── App.tsx        # Componente principal del dashboard
│   ├── main.tsx       # Punto de entrada de React
│   └── index.css      # Estilos globales con Tailwind
├── index.html         # Template HTML
├── vite.config.ts     # Configuración de Vite
├── tsconfig.json      # Configuración de TypeScript
├── tailwind.config.js # Configuración de Tailwind CSS
└── package.json       # Dependencias y scripts
```

## Tecnologías Utilizadas

- **React 19** - Biblioteca UI
- **TypeScript** - Type safety
- **Vite** - Build tool y dev server
- **Recharts** - Biblioteca de gráficos
- **Tailwind CSS** - Framework de estilos utility-first

## Notas

- El proyecto está separado del código Python en el directorio raíz
- Los datos son estáticos y están incluidos en el componente App.tsx
- Para actualizar los datos, modifica las constantes en App.tsx
