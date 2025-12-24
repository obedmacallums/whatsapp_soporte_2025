/**
 * TypeScript Type Definitions for Theme System
 */

export interface ColorScale {
  50: string;
  100: string;
  200: string;
  300: string;
  400: string;
  500: string;
  600: string;
  700: string;
  800: string;
  900: string;
}

export interface GradientConfig {
  from: string;
  to: string;
  via?: string;
  tailwind: string;
}

export type CategoryKey = 'GNSS' | 'SOFTWARE' | 'OTRO' | 'ÓPTICA' | 'OPTICA' | 'COLECTORA' | 'UAS' | 'ESCÁNER' | 'ESCANER' | 'RADIO';
export type ResolutionKey = 'Resuelto' | 'Parcial' | 'Sin Seguimiento' | 'No Resuelto';
export type SatisfactionKey = 'Alta' | 'Media' | 'No Determinable' | 'Baja';
