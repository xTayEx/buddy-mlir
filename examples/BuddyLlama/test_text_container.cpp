#include <buddy/LLM/TextContainer.h>

using namespace buddy;

int main() {
  Text<size_t, 2> textContainer;
  const std::string vocabDir = "../../examples/BuddyLlama/vocab.txt";
  textContainer.loadVocab(vocabDir);
  for (int i = 0; i < 10; i++) {
    textContainer.appendTokenIdx(263);
  }
  std::cout << textContainer.revertLlama() << std::endl;

  return 0;
}
