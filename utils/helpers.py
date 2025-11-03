def format_execution_time(seconds, detailed=False):
    """
    Formata tempo de execução de forma inteligente
    
    Args:
        seconds (float): Tempo em segundos
        detailed (bool): Se retorna versão detalhada
    """
    if detailed:
        # Versão detalhada: "1 hora, 25 minutos e 30 segundos"
        if seconds < 60:
            return f"{seconds:.2f} segundos"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        parts = []
        if hours > 0:
            parts.append(f"{hours} hora{'s' if hours > 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minuto{'s' if minutes > 1 else ''}")
        if secs > 0 or not parts:  # Mostra segundos se for o único ou se > 0
            parts.append(f"{secs:.1f} segundo{'s' if secs != 1 else ''}")
        
        return " e ".join(parts)
    
    else:
        # Versão compacta: "1h 25min 30s"
        if seconds < 60:
            return f"{seconds:.1f}s"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes:02d}min {secs:02.0f}s"
        else:
            return f"{minutes}min {secs:02.0f}s"