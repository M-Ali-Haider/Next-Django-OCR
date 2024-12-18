const Button = ({ Icon, label, className, onClick }) => {
  return (
    <button
      onClick={onClick}
      className={`${className} bg-white border border-[#546fff] rounded-[4px] text-[#546fff] uppercase text-sm font-medium px-8 py-3 active:scale-95 duration-150 transition-all ease-linear`}
    >
      {label}
    </button>
  );
};

export default Button;