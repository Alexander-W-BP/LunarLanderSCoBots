import joblib
import os
import re

class TreeNode:
    def __init__(self, feature=None, threshold=None, operator=None, left=None, right=None, *, value=None):
        """
        Ein Knoten im Entscheidungsbaum.
        """
        self.feature = feature
        self.threshold = threshold
        self.operator = operator  # '<=' oder '>'
        self.left = left
        self.right = right
        self.value = value  # Klassenwert, wenn es ein Blatt ist

    def is_leaf(self):
        return self.value is not None

    def __repr__(self):
        if self.is_leaf():
            return f"Leaf({self.value})"
        else:
            return f"Node({self.feature} {self.operator} {self.threshold})"

class CustomDecisionTree:
    def __init__(self, root):
        """
        Initialisiert den benutzerdefinierten Entscheidungsbaum mit der Wurzel.
        """
        self.root = root

    def predict_sample(self, sample):
        """
        Gibt die Vorhersage für ein einzelnes Datenbeispiel zurück.
        """
        node = self.root
        while not node.is_leaf():
            feature_value = sample.get(node.feature)
            if feature_value is None:
                raise ValueError(f"Feature '{node.feature}' nicht gefunden im Eingabedaten.")
            if node.operator == '<=':
                if feature_value <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            elif node.operator == '>':
                if feature_value > node.threshold:
                    node = node.left
                else:
                    node = node.right
            else:
                raise ValueError(f"Unbekannter Operator '{node.operator}' in Knoten: {node}")
        return node.value

    def predict(self, X):
        """
        Gibt die Vorhersagen für eine Liste von Datenbeispielen zurück.
        """
        return [self.predict_sample(sample) for sample in X]

def parse_tree(lines):
    """
    Parst die Liste von Zeilen aus der Textdatei und baut den Baum auf.
    """
    def parse_node(index, current_indent):
        if index >= len(lines):
            return None, index

        line = lines[index]
        indent = 0
        # Zählen der Anzahl der "|   " am Anfang der Zeile
        while line.startswith("|   "):
            indent += 1
            line = line[4:]

        if indent != current_indent:
            return None, index

        # Entfernt das "|--- " oder "--- " Prefix mit beliebigen Leerzeichen nach dem Operator
        match = re.match(r"(\|--- |--- )(.*)", line)
        if not match:
            raise ValueError(f"Unerwartetes Zeilenformat: {line}")
        line = match.group(2).strip()

        if line.startswith("class:"):
            # Blattknoten
            value = int(line.split(":")[1].strip())
            node = TreeNode(value=value)
            print(f"Geparster Blattknoten: {node}")
            return node, index + 1
        else:
            # Entscheidungsregel
            # Verwenden Sie reguläre Ausdrücke, um die Bedingung zu parsen
            condition_match = re.match(r"(\w+)\s*(<=|>)\s*([-+]?\d*\.\d+|\d+)", line)
            if not condition_match:
                raise ValueError(f"Unerwartetes Bedingungsformat: {line}")
            feature, operator, threshold = condition_match.groups()
            threshold = float(threshold)
            node = TreeNode(feature=feature, threshold=threshold, operator=operator)
            print(f"Geparster interner Knoten: {node} mit Operator: {operator}")
            index += 1
            # Linkes Kind
            node.left, index = parse_node(index, current_indent + 1)
            if node.left is None:
                raise ValueError(f"Linkes Kind fehlt für Knoten: {node}")
            # Rechtes Kind
            node.right, index = parse_node(index, current_indent + 1)
            if node.right is None:
                raise ValueError(f"Rechtes Kind fehlt für Knoten: {node}")
            return node, index

    root, final_index = parse_node(0, 0)
    if final_index != len(lines):
        raise ValueError("Nicht alle Zeilen wurden erfolgreich geparst.")
    return root

def read_tree_from_txt(file_path):
    """
    Liest den Entscheidungsbaum aus einer Textdatei und gibt die Wurzel des Baumes zurück.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip() for line in lines if line.strip() != '']
    print(f"Anzahl der Zeilen zum Parsen: {len(lines)}")
    root = parse_tree(lines)
    return root

def write_tree_to_txt(node, file_path):
    """
    Schreibt den Entscheidungsbaum in eine Textdatei im gewünschten Format.
    """
    lines = []

    def traverse_with_operator(node, depth):
        if node is None:
            raise ValueError(f"Encountered None node at depth {depth}")
        indent = "|   " * depth + "|--- "
        if node.is_leaf():
            lines.append(f"{indent}class: {node.value}")
            print(f"Schreibe Blattknoten: {node.value} bei Tiefe {depth}")
        else:
            condition = f"{node.feature} {node.operator} {node.threshold}"
            lines.append(f"{indent}{condition}")
            print(f"Schreibe internen Knoten: {condition} bei Tiefe {depth}")
            # Linkes Kind
            traverse_with_operator(node.left, depth + 1)
            # Rechtes Kind
            # Bestimmen des Operators für das rechte Kind
            if node.operator == '<=':
                condition_right = f"{node.feature} > {node.threshold}"
            else:
                condition_right = f"{node.feature} <= {node.threshold}"
            indent_right = "|   " * (depth + 1) + "|--- "
            if node.right.is_leaf():
                lines.append(f"{indent_right}class: {node.right.value}")
                print(f"Schreibe Blattknoten: {node.right.value} bei Tiefe {depth +1}")
            else:
                lines.append(f"{indent_right}{condition_right}")
                print(f"Schreibe internen Knoten: {condition_right} bei Tiefe {depth +1}")
                traverse_with_operator(node.right, depth + 1)  # Tiefe +1 statt +2

    traverse_with_operator(node, 0)

    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"Baum erfolgreich in {file_path} geschrieben.")

def main():
    txt_file = 'decision_tree_depth_3.txt'
    joblib_file = 'decision_tree_depth_3.joblib'
    converted_txt_file = 'decision_tree_depth_3_converted.txt'

    # Schritt 1: Überprüfen, ob die ursprüngliche txt-Datei existiert
    if not os.path.exists(txt_file):
        print(f"Die Datei {txt_file} wurde nicht gefunden.")
        return

    # Schritt 2: Lesen der txt-Datei und Erstellen des Baumes
    print(f"Lese den Entscheidungsbaum aus {txt_file}...")
    try:
        root = read_tree_from_txt(txt_file)
    except Exception as e:
        print(f"Fehler beim Parsen der txt-Datei: {e}")
        return

    tree = CustomDecisionTree(root)

    # Schritt 3: Speichern des Baumes als joblib-Datei
    print(f"Speichere den Entscheidungsbaum in {joblib_file}...")
    try:
        joblib.dump(tree, joblib_file)
    except Exception as e:
        print(f"Fehler beim Speichern der joblib-Datei: {e}")
        return
    print("Konvertierung von txt zu joblib abgeschlossen.")

    # Schritt 4: Laden der joblib-Datei
    print(f"Lade den Entscheidungsbaum aus {joblib_file}...")
    try:
        loaded_tree = joblib.load(joblib_file)
    except Exception as e:
        print(f"Fehler beim Laden der joblib-Datei: {e}")
        return
    print("Laden der joblib-Datei erfolgreich.")

    # Schritt 5: Schreiben des Baumes zurück in eine neue txt-Datei
    print(f"Schreibe den Entscheidungsbaum in {converted_txt_file}...")
    try:
        write_tree_to_txt(loaded_tree.root, converted_txt_file)
    except Exception as e:
        print(f"Fehler beim Schreiben der konvertierten txt-Datei: {e}")
        return
    print("Konvertierung von joblib zu txt abgeschlossen.")

    # Optional: Vergleichen der beiden txt-Dateien
    try:
        with open(txt_file, 'r') as original, open(converted_txt_file, 'r') as converted:
            original_lines = [line.rstrip() for line in original]
            converted_lines = [line.rstrip() for line in converted]

        if original_lines == converted_lines:
            print("Die ursprüngliche txt-Datei und die konvertierte txt-Datei sind identisch.")
        else:
            print("Die ursprüngliche txt-Datei und die konvertierte txt-Datei unterscheiden sich.")
    except Exception as e:
        print(f"Fehler beim Vergleichen der txt-Dateien: {e}")

if __name__ == "__main__":
    main()
