import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;

public class Tmp {

  public static void main(String[] args) throws Exception {
    try (FileOutputStream outputStream = new FileOutputStream("./tst", true)) {
      outputStream.write("hello world".getBytes(StandardCharsets.UTF_8));
    }
  }

}
