"""
Filtro per warnings e deprecation messages
"""
import warnings
import os

def suppress_model_warnings():
    """Sopprime i warning noti dai modelli per avere log più puliti"""
    
    # Sopprime il FutureWarning specifico di torch/transformers
    warnings.filterwarnings(
        "ignore", 
        message=".*encoder_attention_mask.*is deprecated.*",
        category=FutureWarning,
        module="torch.*"
    )
    
    # Sopprime altri warning comuni dei modelli
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="transformers.*"
    )
    
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="sentence_transformers.*"
    )
    
    # Sopprime warning di HuggingFace Hub
    warnings.filterwarnings(
        "ignore",
        message=".*clean_up_tokenization_spaces.*",
        category=FutureWarning
    )
    
    # Set environment variables per ridurre verbosity
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    
    print("🔇 Model warnings suppressed for cleaner logs")

def configure_logging_level():
    """Configura il livello di logging per librerie esterne"""
    import logging
    
    # Riduce verbosity di librerie esterne
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)