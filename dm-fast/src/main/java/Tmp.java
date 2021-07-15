import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Tmp {

  public static void main(String[] args) {
    Future<Integer> submit = Executors.newFixedThreadPool(1).submit(() -> 1 / 0);
    try {
      submit.get();
    } catch (Exception e) {
      System.err.println("main thread 抓住了异常");
      e.printStackTrace();
    }

  }
}
