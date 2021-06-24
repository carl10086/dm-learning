import java.time.LocalDateTime;

public class Tmp {

  public static void main(String[] args) throws Exception {
    System.out.println("hour=" + LocalDateTime.now().minusHours(-5L).getHour());
  }

}
