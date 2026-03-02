import importlib

def load_model_class(module_name, model_name):
    try:
        model_module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(f"モジュール '{module_name}' が見つかりません")
    except ImportError as e:
        raise ValueError(f"モジュール '{module_name}' の読み込みに失敗しました: {e}")
    try:
        ModelClass = getattr(model_module, model_name)
    except AttributeError:
        raise ValueError(
            f"{model_name} は {module_name}.py に存在しません"
        )
    return ModelClass