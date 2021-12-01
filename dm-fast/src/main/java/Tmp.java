public class Tmp {

  public static void main(String[] args) throws Exception {
    final long l = System.nanoTime();

    Thread.sleep(1000L);
    System.out.println(System.nanoTime() - l);
  }


  /**
   * 跨线程对象 , 假设有 100 个属性
   */
  private static class CrossThreadObj {

  }
}
