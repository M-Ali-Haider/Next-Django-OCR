import Form from "@/components/Form";
import Header from "@/components/Header";
import Hero from "@/components/Hero/page";

export default function Home() {
  return (
    <>
      <Header />
      <main className="flex flex-col items-center">
        <div className="px-6 max-w-[1200px] w-full flex flex-col items-center mt-12">
          <Hero />
          <Form />
        </div>
      </main>
    </>
  );
}
