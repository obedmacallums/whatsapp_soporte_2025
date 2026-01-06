#!/usr/bin/env python3
"""
WhatsApp Analytics Data Generator
Generates dashboard JSON from CSV data

Supports two CSV formats:
- chats: chats_conversations_2025.csv (JSON ai_analysis column)
- conversations: conversations2_analysis.csv (separate columns)
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from collections import Counter
import unicodedata
import re
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants - Default paths
DEFAULT_CSV_PATH = Path(__file__).parent / 'data' / 'chats_conversations_2025.csv'

# Output paths by format type
OUTPUT_PATHS = {
    'chats': Path(__file__).parent / 'frontend' / 'src' / 'data' / 'soporte-dashboard-data.json',
    'conversations': Path(__file__).parent / 'frontend' / 'src' / 'data' / 'uas-dashboard-data.json'
}

RESOLUTION_MAP = {
    'SI': 'Resuelto',
    'PARCIAL': 'Parcial',
    'NO': 'No Resuelto',
    'SIN_SEGUIMIENTO': 'Sin Seguimiento'
}

SATISFACTION_MAP = {
    'ALTA': 'Alta',
    'MEDIA': 'Media',
    'BAJA': 'Baja',
    'NO_DETERMINABLE': 'No Determinable'
}

PROBLEM_TYPE_MAP = {
    'configuracion': 'Configuración',
    'consulta_tecnica': 'Consulta Técnica',
    'software': 'Software',
    'licencia': 'Licencia',
    'capacitacion': 'Capacitación',
    'hardware': 'Hardware',
    'conectividad': 'Conectividad',
    'otro': 'Otro'
}

MONTH_NAMES = {
    1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
}

# Mapeo manual de productos para consolidación
# Formato: 'variación_normalizada' -> 'nombre_canónico'
PRODUCT_MAPPING = {
    # TBC - Trimble Business Center
    'TBC': 'TBC',
    'TRIMBLE BUSINESS CENTER': 'TBC',
    'TRIMBLE BUSINESS CENTER (TBC)': 'TBC',
    'TBC (TRIMBLE BUSINESS CENTER)': 'TBC',

    # Receptores GNSS - Serie R
    'R8': 'R8',
    'R8S': 'R8',
    'R8-3': 'R8',
    'R10': 'R10',
    'R12': 'R12',
    'R12I': 'R12',

    # Spectra Precision
    'SP60': 'Spectra SP60',
    'SP80': 'Spectra SP80',
    'SPECTRA SP80': 'Spectra SP80',
    'SP85': 'Spectra SP85',
    'SPECTRA SP85': 'Spectra SP85',
    'SP100': 'Spectra SP100',
    'SPECTRA SP100': 'Spectra SP100',
    'SP 100 (SPECTRA)': 'Spectra SP100',
    'SPECTRA': 'Spectra (General)',
    'SPECTRA (RECEPTOR GNSS)': 'Spectra (General)',

    # Trimble Access
    'TRIMBLE ACCESS': 'Trimble Access',

    # Estaciones Totales
    'TRIMBLE S5': 'Trimble S5',
    'TRIMBLE S5/C5': 'Trimble S5',
    'TRIMBLE S7': 'Trimble S7',
    'TRIMBLE S9': 'Trimble S9',

    # Controladores
    'TDC600': 'TDC600',
    'TSC3': 'TSC3',
    'TSC5': 'TSC5',
    'TSC7': 'TSC7',

    # Software adicional
    'SURVEY OFFICE': 'Survey Office',
    'SPSO (SPECTRA PRECISION SURVEY OFFICE)': 'Survey Office',
    'SPECTRA PRECISION SURVEY OFFICE (SPSO)': 'Survey Office',
    'SPECTRA SURVEY OFFICE': 'Survey Office',

    # UAS/Drones
    'UAS': 'UAS',
    'TRIMBLE UAS': 'UAS',

    # Escáneres
    'TX5': 'TX5',
    'TX6': 'TX6',
    'TX8': 'TX8',
    'X7': 'X7',

    # Radios
    'TDL450H': 'TDL450H',
    'TDL450L': 'TDL450L',
    'HPB450': 'HPB450',
    'TRIMTALK': 'TrimTalk',

    # Otros software
    'REALWORKS': 'RealWorks',
    'PATHFINDER OFFICE': 'Pathfinder Office',

    # Genérico
    'GENERAL': 'General',
    'OTRO': 'Otro'
}

# Productos a excluir del gráfico "Productos Más Consultados"
# Útil para filtrar categorías genéricas o poco informativas
EXCLUDED_PRODUCTS = [
    'General',      # Productos sin categoría específica
    'Otro',         # Categoría catch-all
    # Agregar más productos a excluir aquí según necesidad
]


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns:
        Namespace with input, output, and format arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate dashboard JSON from WhatsApp CSV data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default chats format
  python generar_data_graficos.py

  # Use conversations format (auto-detected)
  python generar_data_graficos.py -i data/conversations2_analysis.csv

  # Specify format explicitly
  python generar_data_graficos.py -i data/conversations2_analysis.csv -f conversations

  # Custom output path
  python generar_data_graficos.py -i data/conversations2_analysis.csv -o output/custom.json
        """
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input CSV file (default: data/chats_conversations_2025.csv)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to output JSON file (default: based on format type)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['chats', 'conversations'],
        help='CSV format type: chats (JSON ai_analysis) or conversations (separate columns). Auto-detected if not specified.'
    )
    return parser.parse_args()


def detect_csv_format(csv_path: Path) -> str:
    """
    Detect CSV format by checking column names

    Args:
        csv_path: Path to CSV file

    Returns:
        'chats' if has ai_analysis column (JSON format)
        'conversations' if has ai_provider column (separate columns)

    Raises:
        ValueError: If format cannot be determined
    """
    logger.info(f"Detecting CSV format for {csv_path}")
    df_sample = pd.read_csv(csv_path, nrows=1, encoding='utf-8')
    columns = df_sample.columns.tolist()

    if 'ai_analysis' in columns:
        logger.info("Detected format: chats (JSON ai_analysis)")
        return 'chats'
    elif 'ai_provider' in columns or ('main_question' in columns and 'ai_analysis' not in columns):
        logger.info("Detected format: conversations (separate columns)")
        return 'conversations'
    else:
        raise ValueError(
            f"Unknown CSV format. Expected 'ai_analysis' or 'ai_provider' column. "
            f"Found columns: {columns}"
        )


def normalize_text(text: str) -> str:
    """
    Normaliza texto para comparación y agrupación consistente

    - Elimina tildes/acentos
    - Convierte a mayúsculas
    - Limpia espacios múltiples
    - Remueve caracteres especiales básicos
    """
    if pd.isna(text) or not text:
        return ''

    # Convertir a string
    text = str(text).strip()

    # Eliminar tildes/acentos (NFD = Normalization Form Decomposed)
    text_nfd = unicodedata.normalize('NFD', text)
    text_no_accents = ''.join(
        char for char in text_nfd
        if unicodedata.category(char) != 'Mn'  # Mn = Mark, Nonspacing
    )

    # Mayúsculas
    text_upper = text_no_accents.upper()

    # Limpiar espacios múltiples
    text_clean = re.sub(r'\s+', ' ', text_upper).strip()

    return text_clean


def normalize_product_name(product: str) -> str:
    """
    Normaliza y consolida nombres de productos usando diccionario de mapeo

    Args:
        product: Nombre del producto en cualquier formato

    Returns:
        Nombre canónico del producto o el nombre normalizado si no hay mapeo
    """
    if pd.isna(product) or not product:
        return 'General'

    # Limpiar el producto (tomar solo la primera parte antes de separadores)
    product_clean = str(product).strip()
    for sep in [' (', '/', ',', '-']:
        if sep in product_clean:
            product_clean = product_clean.split(sep)[0].strip()

    # Normalizar el texto (mayúsculas, sin tildes)
    product_normalized = normalize_text(product_clean)

    # Buscar en el diccionario de mapeo
    if product_normalized in PRODUCT_MAPPING:
        return PRODUCT_MAPPING[product_normalized]

    # Si no está en el mapeo, devolver el texto limpio con capitalización
    return product_clean.title()


def load_chats_format(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV in 'chats' format (chats_conversations_2025.csv)
    This format has ai_analysis as a JSON column

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with ai_analysis fields expanded as columns
    """
    logger.info(f"Loading CSV (chats format) from {csv_path}")

    # 1. Load CSV with UTF-8 encoding
    df = pd.read_csv(csv_path, encoding='utf-8')

    # 2. Parse ai_analysis JSON column
    def safe_json_parse(json_str):
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse JSON: {str(json_str)[:50]}...")
            return {
                'category': 'OTRO',
                'subcategory': 'General',
                'problem_type': 'otro',
                'resolution': 'SIN_SEGUIMIENTO',
                'satisfaction': 'NO_DETERMINABLE',
                'agent_name': 'NO_IDENTIFICADO',
                'main_question': '',
                'insights': ''
            }

    # 3. Expand JSON into separate columns
    df['ai_data'] = df['ai_analysis'].apply(safe_json_parse)
    df['category'] = df['ai_data'].apply(lambda x: x.get('category', 'OTRO'))
    df['subcategory'] = df['ai_data'].apply(lambda x: x.get('subcategory', 'General'))
    df['problem_type'] = df['ai_data'].apply(lambda x: x.get('problem_type', 'otro'))
    df['resolution'] = df['ai_data'].apply(lambda x: x.get('resolution', 'SIN_SEGUIMIENTO'))
    df['satisfaction'] = df['ai_data'].apply(lambda x: x.get('satisfaction', 'NO_DETERMINABLE'))
    df['agent_name'] = df['ai_data'].apply(lambda x: x.get('agent_name', 'NO_IDENTIFICADO'))

    # 4. Parse datetime columns
    df['first_message_dt'] = pd.to_datetime(df['first_message'], errors='coerce')
    df['month_num'] = df['first_message_dt'].dt.month

    # 5. Log data quality
    logger.info(f"Loaded {len(df)} conversations")
    logger.info(f"Date parse success: {df['first_message_dt'].notna().sum()}/{len(df)}")
    logger.info(f"Has total_messages: True")

    return df


def load_conversations_format(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV in 'conversations' format (conversations2_analysis.csv)
    This format has separate columns for each analysis field

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame normalized to match chats format structure
    """
    logger.info(f"Loading CSV (conversations format) from {csv_path}")

    # 1. Load CSV with UTF-8 encoding
    df = pd.read_csv(csv_path, encoding='utf-8')

    # 2. Normalize column names to match chats format
    # Use conversation_id as contact_jid equivalent
    df['contact_jid'] = df['conversation_id'].fillna(df.get('conversation', 'unknown'))

    # 3. Create ai_data dict for compatibility with existing functions
    def create_ai_data(row):
        return {
            'main_question': row.get('main_question', ''),
            'category': row.get('category', 'OTRO'),
            'subcategory': row.get('subcategory', 'General'),
            'problem_type': row.get('problem_type', 'otro'),
            'resolution': row.get('resolution', 'SIN_SEGUIMIENTO'),
            'satisfaction': row.get('satisfaction', 'NO_DETERMINABLE'),
            'agent_name': row.get('agent_name', 'NO_IDENTIFICADO'),
            'insights': row.get('insights', '')
        }

    df['ai_data'] = df.apply(create_ai_data, axis=1)

    # 4. Ensure required columns exist (these already exist in this format)
    # Fill missing values with defaults
    df['category'] = df['category'].fillna('OTRO')
    df['subcategory'] = df['subcategory'].fillna('General')
    df['problem_type'] = df['problem_type'].fillna('otro')
    df['resolution'] = df['resolution'].fillna('SIN_SEGUIMIENTO')
    df['satisfaction'] = df['satisfaction'].fillna('NO_DETERMINABLE')
    df['agent_name'] = df['agent_name'].fillna('NO_IDENTIFICADO')

    # 5. This format does NOT have total_messages - set to None/0
    # We don't create a fake total_messages column, functions will check for it
    if 'total_messages' not in df.columns:
        logger.info("Note: 'total_messages' column not present in this format")

    # 6. Parse datetime columns (format is YYYY-MM-DD)
    df['first_message_dt'] = pd.to_datetime(df['first_message'], errors='coerce')
    df['month_num'] = df['first_message_dt'].dt.month

    # 7. Log data quality
    logger.info(f"Loaded {len(df)} conversations")
    logger.info(f"Date parse success: {df['first_message_dt'].notna().sum()}/{len(df)}")
    logger.info(f"Has total_messages: {'total_messages' in df.columns}")

    return df


def load_data(csv_path: Path, csv_format: str) -> pd.DataFrame:
    """
    Load CSV data using the appropriate loader based on format

    Args:
        csv_path: Path to CSV file
        csv_format: 'chats' or 'conversations'

    Returns:
        Normalized DataFrame
    """
    if csv_format == 'chats':
        return load_chats_format(csv_path)
    elif csv_format == 'conversations':
        return load_conversations_format(csv_path)
    else:
        raise ValueError(f"Unknown format: {csv_format}")


def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate 4 key performance indicators

    Note: If 'total_messages' column is not present, message-related KPIs
    will be returned as None (for JSON null serialization)
    """
    total_conversaciones = len(df)

    # Check if total_messages column exists
    has_messages = 'total_messages' in df.columns and df['total_messages'].notna().any()

    if has_messages:
        total_mensajes = int(df['total_messages'].sum())
        promedio_mensajes = round(total_mensajes / total_conversaciones, 1) if total_conversaciones > 0 else 0
        conversacion_mas_larga = int(df['total_messages'].max())
    else:
        # Return None for JSON null serialization
        total_mensajes = None
        promedio_mensajes = None
        conversacion_mas_larga = None

    return {
        'totalConversaciones': total_conversaciones,
        'totalMensajes': total_mensajes,
        'promedioMensajes': promedio_mensajes,
        'conversacionMasLarga': conversacion_mas_larga
    }


def generate_categorias(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate category distribution (no colors - handled in frontend)
    """
    category_counts = df['category'].value_counts().to_dict()

    result = [
        {'name': cat, 'value': int(count)}
        for cat, count in category_counts.items()
    ]

    # Sort by value descending
    result.sort(key=lambda x: x['value'], reverse=True)

    return result


def generate_resolucion(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate resolution data with mapped names and percentages
    """
    # Map values
    df_mapped = df.copy()
    df_mapped['resolution_mapped'] = df_mapped['resolution'].map(RESOLUTION_MAP)

    # Count occurrences
    counts = df_mapped['resolution_mapped'].value_counts().to_dict()
    total = len(df)

    result = [
        {
            'name': name,
            'value': int(count),
            'pct': f"{round(count/total*100)}%"
        }
        for name, count in counts.items()
    ]

    # Sort by value descending
    result.sort(key=lambda x: x['value'], reverse=True)

    return result


def generate_satisfaccion(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate satisfaction level distribution
    """
    df_mapped = df.copy()
    df_mapped['satisfaction_mapped'] = df_mapped['satisfaction'].map(SATISFACTION_MAP)

    counts = df_mapped['satisfaction_mapped'].value_counts().to_dict()

    result = [
        {'name': name, 'value': int(count)}
        for name, count in counts.items()
    ]

    result.sort(key=lambda x: x['value'], reverse=True)

    return result


def generate_tendencia_mensual(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate monthly aggregation of conversations and messages

    Note: If 'total_messages' column is not present, 'mensajes' will be null
    """
    # Filter out rows with invalid dates
    df_valid = df[df['month_num'].notna()].copy()

    # Check if total_messages column exists
    has_messages = 'total_messages' in df.columns and df['total_messages'].notna().any()

    # Group by month
    if has_messages:
        monthly = df_valid.groupby('month_num').agg({
            'contact_jid': 'count',  # conversations
            'total_messages': 'sum'   # messages
        }).reset_index()
        monthly.columns = ['month_num', 'conversaciones', 'mensajes']
    else:
        monthly = df_valid.groupby('month_num').agg({
            'contact_jid': 'count'  # conversations only
        }).reset_index()
        monthly.columns = ['month_num', 'conversaciones']
        monthly['mensajes'] = None

    # Convert to list of dicts with Spanish month names
    result = []
    for _, row in monthly.iterrows():
        month_num = int(row['month_num'])
        mensajes_val = int(row['mensajes']) if pd.notna(row['mensajes']) else None
        result.append({
            'mes': MONTH_NAMES.get(month_num, f"M{month_num}"),
            'conversaciones': int(row['conversaciones']),
            'mensajes': mensajes_val
        })

    # Ensure chronological order
    month_order = list(MONTH_NAMES.values())
    result.sort(key=lambda x: month_order.index(x['mes']) if x['mes'] in month_order else 999)

    return result


def generate_tipos_problema(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate problem type distribution (handling pipe-separated values)
    """
    # Expand pipe-separated problem types
    problem_types = []

    for pt_str in df['problem_type'].dropna():
        # Split by pipe
        types = str(pt_str).split('|')
        problem_types.extend([t.strip() for t in types])

    # Count occurrences
    counts = Counter(problem_types)

    # Map to Spanish labels
    result = [
        {
            'tipo': PROBLEM_TYPE_MAP.get(pt, pt.title()),
            'cantidad': count
        }
        for pt, count in counts.items()
    ]

    # Sort by cantidad descending and take top 7
    result.sort(key=lambda x: x['cantidad'], reverse=True)

    return result[:7]


def generate_productos(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate top products from subcategory field with normalization
    Excludes products listed in EXCLUDED_PRODUCTS
    """
    # Aplicar normalización a todos los productos
    df['product_normalized'] = df['subcategory'].apply(normalize_product_name)

    # Contar ocurrencias
    counts = df['product_normalized'].value_counts().to_dict()

    # Filtrar productos excluidos
    result = [
        {'producto': product, 'consultas': int(count)}
        for product, count in counts.items()
        if product not in EXCLUDED_PRODUCTS
    ]

    # Take top 7
    return result[:7]


def generate_radar_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Calculate performance metrics for radar chart (0-100 scale)
    """
    total = len(df)

    # Resolución %: (SI + PARCIAL) / total
    resueltos = len(df[df['resolution'].isin(['SI', 'PARCIAL'])])
    resolucion_pct = round((resueltos / total) * 100)

    # Satisfacción %: (ALTA + MEDIA) / total
    satisfechos = len(df[df['satisfaction'].isin(['ALTA', 'MEDIA'])])
    satisfaccion_pct = round((satisfechos / total) * 100)

    # Seguimiento %: cases with follow-up (not SIN_SEGUIMIENTO)
    con_seguimiento = len(df[df['resolution'] != 'SIN_SEGUIMIENTO'])
    seguimiento_pct = round((con_seguimiento / total) * 100)

    return [
        {'subject': 'Resolución', 'A': resolucion_pct, 'fullMark': 100},
        {'subject': 'Satisfacción', 'A': satisfaccion_pct, 'fullMark': 100},
        {'subject': 'Seguimiento', 'A': seguimiento_pct, 'fullMark': 100}
    ]


def select_top_conversations(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
    """
    Select top N high-satisfaction conversations using multi-criteria scoring

    Criteria:
    - satisfaction == "ALTA" (required)
    - Insights length (weight: 0.5 when no messages, 0.4 otherwise)
    - Category diversity (weight: 0.3)
    - Resolution quality (weight: 0.2)
    - Message count in sweet spot 50-300 (weight: 0.1, only if available)

    Note: If 'total_messages' is not available, scoring uses only insights and resolution
    """
    # Filter to ALTA satisfaction AND SI resolution
    high_sat = df[(df['satisfaction'] == 'ALTA') & (df['resolution'] == 'SI')].copy()

    if len(high_sat) == 0:
        logger.warning("No high satisfaction conversations found")
        return []

    # Extract insights from ai_data
    high_sat['insights'] = high_sat['ai_data'].apply(
        lambda x: x.get('insights', '')
    )
    high_sat['insights_len'] = high_sat['insights'].str.len()

    # Filter out conversations with very short insights (< 50 chars)
    high_sat = high_sat[high_sat['insights_len'] >= 50].copy()

    if len(high_sat) == 0:
        logger.warning("No high satisfaction conversations with sufficient insights")
        return []

    # Check if total_messages column exists
    has_messages = 'total_messages' in df.columns and df['total_messages'].notna().any()

    # Normalize scores (0-1 scale)
    max_insights = high_sat['insights_len'].max()
    high_sat['insights_score'] = high_sat['insights_len'] / max_insights if max_insights > 0 else 0

    # Resolution score (SI=1.0, PARCIAL=0.6, NO=0.3, SIN_SEGUIMIENTO=0)
    resolution_scores = {'SI': 1.0, 'PARCIAL': 0.6, 'NO': 0.3, 'SIN_SEGUIMIENTO': 0}
    high_sat['resolution_score'] = high_sat['resolution'].map(resolution_scores).fillna(0)

    # Message count score (only if available)
    if has_messages:
        high_sat['msg_score'] = high_sat['total_messages'].apply(
            lambda x: 1.0 if 50 <= x <= 300 else max(0, 1 - abs(x - 175) / 500)
        )
        # Calculate weighted total score with message component
        high_sat['total_score'] = (
            high_sat['insights_score'] * 0.4 +
            high_sat['resolution_score'] * 0.2 +
            high_sat['msg_score'] * 0.1
        )
    else:
        # Calculate weighted total score without message component
        high_sat['total_score'] = (
            high_sat['insights_score'] * 0.5 +
            high_sat['resolution_score'] * 0.3
        )

    # Sort by score and select top N ensuring category diversity
    selected = []
    categories_used = set()

    # First pass: top scorers from different categories
    for _, row in high_sat.sort_values('total_score', ascending=False).iterrows():
        cat = row['category']
        if cat not in categories_used and len(selected) < n:
            selected.append(row)
            categories_used.add(cat)

    # Second pass: fill remaining slots with highest scores
    remaining = n - len(selected)
    if remaining > 0:
        already_selected = {row['contact_jid'] for row in selected}
        for _, row in high_sat.sort_values('total_score', ascending=False).iterrows():
            if row['contact_jid'] not in already_selected:
                selected.append(row)
                if len(selected) >= n:
                    break

    # Format output
    result = []
    for row in selected:
        ai_data = row['ai_data']
        # Handle totalMessages - use None if not available
        total_messages_val = int(row['total_messages']) if has_messages else None
        result.append({
            'id': row['contact_jid'],
            'category': row['category'],
            'mainQuestion': ai_data.get('main_question', ''),
            'insights': ai_data.get('insights', ''),
            'subcategory': row['subcategory'],
            'totalMessages': total_messages_val
        })

    logger.info(f"Selected {len(result)} conversation examples from {len(high_sat)} candidates")

    return result


def main():
    """
    Main execution: load data, generate all datasets, write JSON

    Supports two CSV formats via command line arguments:
    - chats: chats_conversations_2025.csv (JSON ai_analysis)
    - conversations: conversations2_analysis.csv (separate columns)
    """
    # Parse command line arguments
    args = parse_args()

    logger.info("Starting data generation...")

    # 1. Determine input path
    csv_path = Path(args.input) if args.input else DEFAULT_CSV_PATH
    logger.info(f"Input file: {csv_path}")

    # 2. Detect or use specified format
    if args.format:
        csv_format = args.format
        logger.info(f"Using specified format: {csv_format}")
    else:
        csv_format = detect_csv_format(csv_path)

    # 3. Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_PATHS[csv_format]
    logger.info(f"Output file: {output_path}")

    # 4. Load and prepare data
    df = load_data(csv_path, csv_format)

    # 5. Generate all datasets
    dashboard_data = {
        'kpis': calculate_kpis(df),
        'categorias': generate_categorias(df),
        'resolucion': generate_resolucion(df),
        'satisfaccion': generate_satisfaccion(df),
        'tendenciaMensual': generate_tendencia_mensual(df),
        'tiposProblema': generate_tipos_problema(df),
        'productos': generate_productos(df),
        'radarData': generate_radar_data(df),
        'conversationExamples': select_top_conversations(df, n=5)
    }

    # 6. Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 7. Write JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)

    # 8. Log summary
    logger.info(f"✓ Successfully generated {output_path}")
    logger.info(f"  - Format: {csv_format}")
    logger.info(f"  - {dashboard_data['kpis']['totalConversaciones']} conversations")

    # Handle None values in logging for message stats
    total_msgs = dashboard_data['kpis']['totalMensajes']
    if total_msgs is not None:
        logger.info(f"  - {total_msgs} messages")
    else:
        logger.info(f"  - messages: N/A (not in source format)")

    logger.info(f"  - {len(dashboard_data['categorias'])} categories")
    logger.info(f"  - {len(dashboard_data['productos'])} products (consolidated)")
    logger.info(f"  - {len(dashboard_data['tendenciaMensual'])} months")
    logger.info(f"  - {len(dashboard_data['conversationExamples'])} conversation examples")


if __name__ == '__main__':
    main()
