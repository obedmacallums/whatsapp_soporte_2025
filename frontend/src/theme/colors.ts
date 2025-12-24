/**
 * Professional Color Palette System
 * Pastel-based theme with soft gradients for corporate dashboards
 */

// Base Palettes - Complete color scales (50-900)
export const basePalette = {
  // Gris-Azul (Primary) - Suave y profesional
  grayBlue: {
    50: '#f0f4f8',
    100: '#d9e4ed',
    200: '#b8cde0',
    300: '#92b3d3',
    400: '#7099c6',
    500: '#567fa8',   // Main
    600: '#3f5f7e',
    700: '#2d4356',
    800: '#1e2d3a',
    900: '#0f1619',
  },

  // Lavanda (Secondary) - Elegante
  lavender: {
    50: '#f7f5fb',
    100: '#ebe6f4',
    200: '#d8cfe9',
    300: '#c1b3db',
    400: '#a896cd',
    500: '#8f7abe',   // Main
    600: '#6d5a96',
    700: '#4d3f6b',
    800: '#322847',
    900: '#1a1425',
  },

  // Menta (Success) - Fresco y positivo
  mint: {
    50: '#f0faf7',
    100: '#daf2eb',
    200: '#b5e5d8',
    300: '#8fd3c0',
    400: '#6bc0a8',
    500: '#4ca890',   // Main
    600: '#3a8270',
    700: '#295c51',
    800: '#1b3b35',
    900: '#0e1e1b',
  },

  // Durazno/Peach (Accent) - Cálido
  peach: {
    50: '#fdf6f3',
    100: '#f9e9e1',
    200: '#f3d3c3',
    300: '#ebb8a0',
    400: '#e19c7d',
    500: '#d1805f',   // Main
    600: '#a66248',
    700: '#754533',
    800: '#4a2c21',
    900: '#251612',
  },

  // Terracota (Error) - Profesional, no agresivo
  terracotta: {
    50: '#fdf4f3',
    100: '#f8e3df',
    200: '#f1c7bf',
    300: '#e7a69a',
    400: '#d97070',   // Main (requirement)
    500: '#c85a54',   // Deeper (requirement)
    600: '#a04741',
    700: '#73322e',
    800: '#49201e',
    900: '#251010',
  },

  // Ámbar (Warning) - Suave
  amber: {
    50: '#fef9f3',
    100: '#fdefe1',
    200: '#faddbd',
    300: '#f7c794',
    400: '#f3af69',
    500: '#e89641',   // Main
    600: '#c27930',
    700: '#8a5622',
    800: '#563616',
    900: '#2b1b0b',
  },

  // Neutrales (Negro, grises, blanco)
  neutral: {
    white: '#ffffff',
    50: '#f8f9fa',
    100: '#f1f3f5',
    200: '#e9ecef',
    300: '#dee2e6',
    400: '#ced4da',
    500: '#adb5bd',
    600: '#6c757d',
    700: '#495057',
    800: '#343a40',
    900: '#212529',
    black: '#000000',
  },
};

// Data Colors - For chart visualizations
export const dataColors = {
  categories: {
    GNSS: basePalette.grayBlue[400],
    SOFTWARE: basePalette.lavender[400],
    OTRO: basePalette.lavender[300],
    'ÓPTICA': basePalette.lavender[200],
    OPTICA: basePalette.lavender[200],
    COLECTORA: basePalette.grayBlue[300],
    UAS: basePalette.mint[400],
    'ESCÁNER': basePalette.mint[500],
    ESCANER: basePalette.mint[500],
    RADIO: basePalette.mint[300],
    fallback: basePalette.neutral[400],
  },

  resolution: {
    'Resuelto': basePalette.mint[500],
    'Parcial': basePalette.amber[500],
    'Sin Seguimiento': basePalette.neutral[500],
    'No Resuelto': basePalette.terracotta[400],
    fallback: basePalette.neutral[400],
  },

  satisfaction: {
    'Alta': basePalette.mint[500],
    'Media': basePalette.amber[500],
    'No Determinable': basePalette.neutral[500],
    'Baja': basePalette.terracotta[400],
    fallback: basePalette.neutral[400],
  },

  // Para series en charts multi-línea
  series: [
    basePalette.grayBlue[400],
    basePalette.lavender[400],
    basePalette.mint[400],
    basePalette.peach[400],
  ],
};

// Chart Colors - Grid, axes, tooltips
export const chartColors = {
  grid: basePalette.neutral[200],
  axis: basePalette.neutral[500],
  axisLabel: basePalette.neutral[600],
  tooltip: {
    background: basePalette.neutral[900],
    text: basePalette.neutral.white,
    border: basePalette.neutral[700],
  },
};

// Gradients - Soft professional gradients
export const gradients = {
  kpi: {
    primary: {
      from: basePalette.grayBlue[600],
      to: basePalette.grayBlue[700],
      tailwind: 'bg-gradient-to-br from-grayBlue-600 to-grayBlue-700',
    },
    secondary: {
      from: basePalette.lavender[600],
      to: basePalette.lavender[700],
      tailwind: 'bg-gradient-to-br from-lavender-600 to-lavender-700',
    },
    accent: {
      from: basePalette.mint[600],
      to: basePalette.mint[700],
      tailwind: 'bg-gradient-to-br from-mint-600 to-mint-700',
    },
    info: {
      from: basePalette.peach[600],
      to: basePalette.peach[700],
      tailwind: 'bg-gradient-to-br from-peach-600 to-peach-700',
    },
  },

  header: {
    from: basePalette.neutral.black,
    via: basePalette.neutral[900],
    to: basePalette.grayBlue[900],
    tailwind: 'bg-gradient-to-r from-black via-neutral-900 to-grayBlue-900',
  },

  background: {
    from: basePalette.neutral[50],
    via: basePalette.neutral[100],
    to: basePalette.grayBlue[50],
    tailwind: 'bg-gradient-to-br from-neutral-50 via-neutral-100 to-grayBlue-50',
  },

  chartBar: {
    primary: [
      { offset: '0%', color: basePalette.grayBlue[400] },
      { offset: '100%', color: basePalette.lavender[400] },
    ],
    secondary: [
      { offset: '0%', color: basePalette.mint[400] },
      { offset: '100%', color: basePalette.grayBlue[400] },
    ],
  },
};

// Utility Functions - Type-safe color getters
export const getCategoryColor = (category: string): string => {
  return dataColors.categories[category as keyof typeof dataColors.categories]
    || dataColors.categories.fallback;
};

export const getResolutionColor = (status: string): string => {
  return dataColors.resolution[status as keyof typeof dataColors.resolution]
    || dataColors.resolution.fallback;
};

export const getSatisfactionColor = (level: string): string => {
  return dataColors.satisfaction[level as keyof typeof dataColors.satisfaction]
    || dataColors.satisfaction.fallback;
};
