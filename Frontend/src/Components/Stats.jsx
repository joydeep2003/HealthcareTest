import React from 'react';

function Stats() {
return (
    <div>
    <div className='flex flex-col items-center justify-center background bg-transparent p-3 background'>
        <div className='ml-10 w-[1080px] relative'>
            <div className='absolute inset-0 backdrop-filter backdrop-blur-md bg-transparent flex flex-row'></div>
            <div className='relative z-10 text-white p-2 items-center text-center'>
                <div className='text-2xl'>
                    Similar symptoms don't necessarily mean everyone is infected with the same disease.<br/> However, final testings are required for proper medication.
                </div>
                <div className='text-2xl'>
                    Machine learning and interpretability can help doctors make better decisions.
                </div>
            </div>
            <div className='relative z-10 text-white p-2 items-center text-center mb-3 mt-2'>
                <div className='flex justify-around'>
                    <img className='h-[576px] w-[964px] rounded' src="/patientsbed.jpeg" alt="HealthLens Image" />
                </div>  
            </div>
        </div>
    </div>
    <div className='flex flex-col items-center justify-center background bg-transparent p-3 background'>
        <div className='ml-10 w-[1080px] relative'>
            <div className='absolute inset-0 backdrop-filter backdrop-blur-md bg-transparent flex flex-row'></div>
            <div className='relative z-10 text-white p-2 items-center text-center'>
                <div className='text-2xl'>
                The image provides information about the number of deaths from pneumonia in 2019 <br/>, categorized by different age groups.
                </div>
                <div className='text-2xl'>
                This infographic emphasizes the significant impact of pneumonia across various age <br/> ranges, with children under five being particularly affected.
                </div>
            </div>
            <div className='relative z-10 text-white p-2 items-center text-center mb-3 mt-2'>
                <div className='flex justify-around'>
                    <img className='h-[650px] w-[964px] rounded' src="/img2.jpeg" alt="HealthLens Image" />
                </div>  
            </div>
        </div>
    </div>
    <div className='flex flex-col items-center justify-center background bg-transparent p-3 background'>
        <div className='ml-10 w-[1080px] relative'>
            <div className='absolute inset-0 backdrop-filter backdrop-blur-sm bg-transparent flex flex-row'></div>
            {/* <div className='relative z-10 text-white p-2 items-center text-center'>
                <div className='text-2xl'>
                    Similar symptoms don't necessarily mean everyone is infected with the same disease.<br/> However, final testings are required for proper medication.
                </div>
                <div className='text-2xl'>
                    Machine learning and interpretability can help doctors make better decisions.
                </div>
            </div> */}
            <div className='relative z-10 text-white p-2 items-center text-center mb-3 mt-2'>
                <div className='flex justify-around'>
                    <img className='h-[576px] w-[980px] rounded' src="/img3.jpeg" alt="HealthLens Image" />
                </div>  
            </div>
        </div>
    </div>
    </div>
);
}

export default Stats;
