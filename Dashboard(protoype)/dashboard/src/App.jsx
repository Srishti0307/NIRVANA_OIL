import image0 from "./images/image0.png";
import image1 from "./images/image1.png";
import image2 from "./images/image2.png";
import image3 from "./images/image3.png";
import image4 from "./images/image4.png";
import oilspill1 from "./images/oilspill1.png";
import oilspill2 from "./images/oilspill2.png";
import oilspill3 from "./images/oilspill3.png";
import React, { useEffect } from "react";
import ReactDOM from "react-dom/client";
import "./style.css";

import runAnimations, { allLinks, allFunctions } from "./scripts";
const App = () => {
  useEffect(() => {
    runAnimations();
  }, []);

  const newFunction = (param1, param2, param3, param4,param5,param6)=>{
    document.querySelector('.c-3977123-9898-0').innerHTML = param1;
    document.querySelector('.c-779875-98135-0 ').innerHTML = param2;
    document.querySelector('.c-1234567-98129-0 ').innerHTML = param3;
    document.querySelector('.c-73-98123-0  ').innerHTML = param4;
    document.querySelector('.c-58kmsq-appx-98142-0  ').innerHTML = param5;
    document.querySelector('.oil-tanker-98147-0  ').innerHTML = param6;
  }

  const addImage = (param) =>{
    document.querySelector('.image-23-2027').innerHTML = '';
    document.querySelector('.image-23-2027').innerHTML = 
    `<img src =${param} className="pos-abs  image-div bg-no-repeat fill-parent bg-cover nodeBg-2027 " alt="image 231 Image" height = "100%", width = "100%">`
  }

  return (
    <div className="parent-div ">
      <div className=" dashboard-16127 pos-abs">
        {/* Left Side Menu1 */}
        <section className=" left-side-menu-16128 pos-abs">
          <div className=" menu-bg-16129 pos-abs"></div>
          <div className=" user-profile-16190 pos-abs">
            <div className=" image-32-8230 pos-abs">
              <img
                src={image0}
                className="pos-abs  image-div bg-no-repeat fill-parent bg-cover nodeBg-8230 "
                alt="image 321 Image"
              />{" "}
            </div>
            <div className=" frame-80-16192 pos-abs">
              <div className=" name-16193 pos-abs">
                <span className="name-16193-0 ">{`Raj Kr. Sharma`}</span>
              </div>
              <div className=" tittle-16194 pos-abs">
                <span className="tittle-16194-0 ">Coastal Operations Director</span>
              </div>
            </div>
          </div>
          <div className=" options-16130 pos-abs">
            <div className=" frame-25-16131 pos-abs">
              <div className=" frame-24-16132 pos-abs">
                <div className=" frame-19-16133 pos-abs">
                  <div className=" frame-9-16134 pos-abs">
                    <div className=" template-16135 pos-abs">
                      <div className=" vector-16136 pos-abs">
                      </div>
                    </div>
                    <div className=" dashboard-16137 pos-abs" onClick={() => window.location.reload()}
                    style={{ cursor: 'pointer' }}>
                      <span className="dashboard-16137-0 ">Dashboard</span>
                    </div>
                  </div>
                </div>
              </div>
              <div className=" discover-16139 pos-abs">
                <div className=" frame-1-16140 pos-abs">
                  <div className=" homeoutline-16141 pos-abs">
                    <div className=" vector-16142 pos-abs">
                    </div>
                  </div>
                  <div className=" incident-histor-16143 pos-abs">
                    <span className="incident-histor-16143-0 " href="#" style={{ cursor: 'pointer'}}>{`Incident History`}</span>
                  </div>
                </div>
              </div>
              <div className=" notifications-16164 pos-abs">
                <div className=" frame-23-16165 pos-abs">
                  <div className=" frame-6-16166 pos-abs">
                    <div className=" belloutline-16167 pos-abs">
                      <div className=" vector-16168 pos-abs">
                      </div>
                    </div>
                    <div className=" notifications-16169 pos-abs">
                      <span className="notifications-16169-0 " href="#" style={{ cursor: 'pointer'}}>{`Notifications`}</span>
                    </div>
                  </div>
                  <div className=" frame-20-16170 pos-abs">
                    <div className=" ellipse-14-16171 pos-abs"></div>
                    <div className=" c-5-16172 pos-abs">
                      <span className="c-5-16172-0 ">{`5`}</span>
                    </div>
                  </div>
                </div>
              </div>
              <div className=" settings-16173 pos-abs">
                <div className=" frame-3-16174 pos-abs">
                  <div className=" icon-16175 pos-abs">
                    <div className=" vector-16176 pos-abs">
                      <div className="nodeBg-16176 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className=" frame-26-16179 pos-abs">
              <div className=" frame-17-16180 pos-abs">
                <div className=" frame-7-16181 pos-abs">
                  <div className=" shieldexclamati-16182 pos-abs">
                    <div className=" vector-16183 pos-abs">
                      <div className="nodeBg-16183 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
                    </div>
                  </div>
                  <div className=" help--support-16184 pos-abs">
                    <span className="help--support-16184-0 " href="#" style={{ cursor: 'pointer'}}>{`Help & Support`}</span>
                  </div>
                </div>
              </div>
              <div className=" frame-18-16185 pos-abs">
                <div className=" frame-8-16186 pos-abs">
                  <div className=" uploadoutline-16187 pos-abs">
                    <div className=" vector-16188 pos-abs">
                      <div className="nodeBg-16188 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
                    </div>
                  </div>
                  <div className=" log-out-16189 pos-abs">
                  <a className="log-out-16189-0 " href="http://localhost:5174" rel="noopener noreferrer">Log Out</a>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
        {/* All Contents1 */}
        <section className=" all-contents-16262 pos-abs">
          <div className=" user-panel-85146 pos-abs">
            <div className=" rectangle-15-85147 pos-abs"></div>
            <div className=" frame-44-85149 pos-abs">
              <div className=" mingcuteleftfil-85151 pos-abs">
                <div className=" group-I85151_6967 pos-abs">
                  <div className=" vector-I85151_6968 pos-abs">
                    <div className="nodeBg-I85151_6968 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
                  </div>
                  <div className=" vector-I85151_6969 pos-abs">
                    <div className="nodeBg-I85151_6969 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
                  </div>
                </div>
              </div>
            </div>
            <div className=" oil-spill-alert-85148 pos-abs">
              <span className="oil-spill-alert-85148-0 ">{`Oil spill alerts`}</span>
            </div>
            <div className=" frame-60-85154 pos-abs" onClick=
            {()=>
            {
              newFunction(28.5099,-86.1433,9744567,'46%','25kmsq approx','Pipeline leak');
              addImage(oilspill1);
              }}>
              <div className=" frame-52-85155 pos-abs">
                <div className=" frame-46-85157 pos-abs">
                  <div className=" silver-admin-da-85158 pos-abs">
                    <span className="silver-admin-da-85158-0 ">{`Silver- Admin dashboard`}</span>
                  </div>
                  <div className=" index-html-85159 pos-abs">
                    <span className="index-html-85159-0 ">{`Index. html`}</span>
                  </div>
                </div>
                <div className=" image-36-9035 pos-abs">
                  <img
                    src={image1}
                    className="pos-abs  image-div bg-no-repeat fill-parent bg-cover nodeBg-9035 "
                    alt="image 361 Image"
                  />{" "}
                </div>
              </div>
              <div className=" frame-55-85160 pos-abs">
                <div className=" rectangle-35-85161 pos-abs"></div>
                <div className=" c-228-85162 pos-abs">
                  <span className="c-228-85162-0 ">{`228`}</span>
                </div>
              </div>
            </div>
            <div className=" line-7-85153 pos-abs"></div>
            <div className=" frame-59-85163 pos-abs">
              <div className=" frame-51-85164 pos-abs">
                <div className=" frame-47-85166 pos-abs" onClick={()=>{
                  newFunction(28.8374,-91.5031,9812345,'92%','40kmsq approx','Equipment Failure');
                  addImage(oilspill2);
                }}>
                  <div className=" dimond-admin-da-85167 pos-abs">
                    <span className="dimond-admin-da-85167-0 ">{`Dimond- Admin dashboard`}</span>
                  </div>
                  <div className=" index-html-85168 pos-abs">
                    <span className="index-html-85168-0 ">{`Index. html`}</span>
                  </div>
                </div>
                <div className=" image-37-9036 pos-abs">
                  <img
                    src={image1}
                    className="pos-abs  image-div bg-no-repeat fill-parent bg-cover nodeBg-9036 "
                    alt="image 371 Image"
                  />{" "}
                </div>
              </div>
              <div className=" frame-56-85169 pos-abs">
                <div className=" rectangle-35-85170 pos-abs"></div>
                <div className=" c-150-85171 pos-abs">
                  <span className="c-150-85171-0 ">{`150`}</span>
                </div>
              </div>
            </div>
            <div className=" line-6-85152 pos-abs"></div>
            <div className=" frame-58-85172 pos-abs" onClick={()=>{
              newFunction(24.132,-83.1317,9876543,'87%','30kmsq approx','Tanker Grounding');
              addImage(oilspill3);
            }}>
              <div className=" frame-53-85173 pos-abs">
                <div className=" frame-45-85175 pos-abs">
                  <div className=" gold-admin-dash-85176 pos-abs">
                    <span className="gold-admin-dash-85176-0 ">{`Gold- Admin dashboard`}</span>
                  </div>
                  <div className=" index-html-85177 pos-abs">
                    <span className="index-html-85177-0 ">{`Index. html`}</span>
                  </div>
                </div>
                <div className=" image-38-9037 pos-abs">
                  <img
                    src={image1}
                    className="pos-abs  image-div bg-no-repeat fill-parent bg-cover nodeBg-9037 "
                    alt="image 381 Image"
                  />{" "}
                </div>
              </div>
              <div className=" frame-57-85178 pos-abs">
                <div className=" rectangle-35-85179 pos-abs"></div>
                <div className=" c-180-85180 pos-abs">
                  <span className="c-180-85180-0 ">{`180`}</span>
                </div>
              </div>
            </div>
          </div>
          <div className=" group-8-98131 pos-abs">
            <div className=" rectangle-17-98132 pos-abs"></div>
            <div className=" frame-25-98133 pos-abs">
              <div className=" longitude-98134 pos-abs">
                <span className="longitude-98134-0 ">{`Longitude`}</span>
              </div>
              <div className=" c-779875-98135 pos-abs">
                <span className="c-779875-98135-0 ">{`-77.9875`}</span>
              </div>
              <div className=" c-26-new-session--98136 pos-abs">
                <span className="c-26-new-session--98136-0 ">{`+26%`}</span>
                <span className="c-26-new-session--98136-1 ">{` new session today`}</span>
              </div>
            </div>
          </div>
          <div className=" group-5-9894 pos-abs">
            <div className=" rectangle-17-9895 pos-abs"></div>
            <div className=" frame-25-9896 pos-abs">
              <div className=" latitude-9897 pos-abs">
                <span className="latitude-9897-0 ">{`Latitude`}</span>
              </div>
              <div className=" c-3977123-9898 pos-abs">
                <span className="c-3977123-9898-0 ">{`39.77123`}</span>
              </div>
            </div>
          </div>
          <div className=" group-7-98125 pos-abs">
            <div className=" rectangle-17-98126 pos-abs"></div>
            <div className=" rectangle-18-98137 pos-abs"></div>
            <div className=" frame-25-98127 pos-abs">
              <div className=" imo-number-98128 pos-abs">
                <span className="imo-number-98128-0 ">{`IMO Number`}</span>
              </div>
              <div className=" c-1234567-98129 pos-abs">
                <span className="c-1234567-98129-0 ">{`1234567`}</span>
              </div>
              <div className=" c-26-new-session--98130 pos-abs">
                <span className="c-26-new-session--98130-0 ">{`+26%`}</span>
                <span className="c-26-new-session--98130-1 ">{` new session today`}</span>
              </div>
            </div>
          </div>
          <div className=" group-6-98119 pos-abs">
            <div className=" rectangle-17-98120 pos-abs"></div>
            <div className=" frame-25-98121 pos-abs">
              <div className=" spill-rate-98122 pos-abs">
                <span className="spill-rate-98122-0 ">{`Spill Rate`}</span>
              </div>
              <div className=" c-73-98123 pos-abs">
                <span className="c-73-98123-0 ">{`73%`}</span>
              </div>
              <div className=" c-26-new-session--98124 pos-abs">
                <span className="c-26-new-session--98124-0 ">{`+26%`}</span>
                <span className="c-26-new-session--98124-1 ">{` new session today`}</span>
              </div>
            </div>
          </div>
          <div className=" group-9-98138 pos-abs">
            <div className=" rectangle-17-98139 pos-abs"></div>
            <div className=" group-5-98143 pos-abs">
              <div className=" rectangle-17-98144 pos-abs"></div>
              <div className=" frame-25-98145 pos-abs">
                <div className=" possible-cause-98146 pos-abs">
                  <span className="possible-cause-98146-0 ">{`Possible cause`}</span>
                </div>
                <div className=" oil-tanker-98147 pos-abs">
                  <span className="oil-tanker-98147-0 ">{`Oil Tanker `}</span>
                </div>
              </div>
            </div>
            <div className=" frame-25-98140 pos-abs">
              <div className=" spread-area-98141 pos-abs">
                <span className="spread-area-98141-0 ">{`Spread  Area`}</span>
              </div>
              <div className=" c-58kmsq-appx-98142 pos-abs">
                <span className="c-58kmsq-appx-98142-0 ">{`58kmsq appx`}</span>
              </div>
            </div>
          </div>
          <div className=" frame-21-10155 pos-abs">
            <div className=" generate-report-10157 pos-abs">
              <span className="generate-report-10157-0 ">{`Generate report`}</span>
            </div>
          </div>
        </section>
        {/* Navbar1 */}
        <section className=" navbar-9856 pos-abs">
          <div className=" group-4-9857 pos-abs">
            <div className=" group-3-9858 pos-abs">
              <div className=" rectangle-16-9859 pos-abs"></div>
              <div className=" frame-24-9860 pos-abs">
                <div className=" octiconsearch16-9861 pos-abs">
                  <div className=" vector-I9861_1114 pos-abs">
                    <div className="nodeBg-I9861_1114 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
                  </div>
                </div>
                <div className="search-any-thin-9862 pos-abs">
                  <input
                    type="text"
                    className="search-any-thin-9862-0"
                    placeholder="Search anything here"
                  />
                </div>
              </div>
            </div>
            <div className=" frame-23-9863 pos-abs">
              <div className=" frame-19-9864 pos-abs">
                <div className=" phsealwarning-9865 pos-abs">
                  <div className=" vector-I9865_1109 pos-abs">
                    <div className="nodeBg-I9865_1109 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
                  </div>
                </div>
                <div className="need-help-9866 pos-abs">
                  <a href="#" className="need-help-9866-0">Get Help</a>
                </div>
              </div>
              <div className=" frame-20-9867 pos-abs">
                <div className=" carbonsettings-9868 pos-abs">
                  <div className=" vector-I9868_1111 pos-abs">
                    <div className="nodeBg-I9868_1111 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
                  </div>
                  <div className=" vector-I9868_1112 pos-abs">
                    <div className="nodeBg-I9868_1112 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
                  </div>
                </div>
                <div className="settings-9869 pos-abs">
                  <a href="#" className="settings-9869-0" style={{ cursor: 'pointer'}}>Settings</a>
                </div>
              </div>
              <div className=" frame-22-9870 pos-abs" onClick={() => {
                const url = window.location.href;
                navigator.clipboard.writeText(url);
                alert(`Page URL copied: ${url}`);
                }}
                style={{ cursor: 'pointer' }} /* Makes the div look clickable */>
                <div className=" icons8share-9871 pos-abs">
                  <div className=" vector-I9871_1107 pos-abs">
                  <div className="nodeBg-I9871_1107 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat"></div>
                  </div>
                </div>
                <div className=" share-9872 pos-abs">
                  <span className="share-9872-0 ">{`Share`}</span>
                </div>
              </div>
              <div className=" frame-21-9873 pos-abs" onClick={() => {
                alert(`Data exported successfully!`);
                }}
                style={{ cursor: 'pointer' }}>
                <div className=" iconparkoutline-9874 pos-abs">
                  <div className=" group-I9874_1102 pos-abs">
                    <div className=" vector-I9874_1105 pos-abs">
                      <div className="nodeBg-I9874_1105 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
                    </div>
                    <div className=" vector-I9874_1103 pos-abs">
                      <div className="nodeBg-I9874_1103 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
                    </div>
                    <div className=" vector-I9874_1104 pos-abs">
                      <div className="nodeBg-I9874_1104 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
                    </div>
                  </div>
                </div>
                <div className=" data-export-9875 pos-abs">
                  <span className="data-export-9875-0 ">{`Export data`}</span>
                </div>
              </div>
            </div>
          </div>
        </section>
        {/* Frame 221 */}
        <section className=" frame-22-10151 pos-abs" onClick={() => {
                alert(`Thanks for reporting!`);
                }}
                style={{ cursor: 'pointer' }}>
          <div className=" report-incident-10153 pos-abs">
            <span className="report-incident-10153-0 ">Report Incident</span>
          </div>
        </section>
        {/* Group 26091 */}
        <section className=" group-2609-9885 pos-abs">
          <div className=" user-panel-8557 pos-abs">
            <div className=" rectangle-15-8558 pos-abs"></div>
            <div className=" frame-44-8560 pos-abs">
              <div className=" mingcuteleftfil-8562 pos-abs">
                <div className=" group-I8562_6967 pos-abs">
                  <div className=" vector-I8562_6968 pos-abs">
                    <div className="nodeBg-I8562_6968 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
                  </div>
                  <div className=" vector-I8562_6969 pos-abs">
                    <div className="nodeBg-I8562_6969 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
                  </div>
                </div>
              </div>
            </div>
            <div className=" ais-alerts-8559 pos-abs">
              <span className="ais-alerts-8559-0 ">{`AIS alerts`}</span>
            </div>
            <div className=" frame-60-8565 pos-abs">
              <div className=" frame-52-8566 pos-abs">
                <div className=" image-33-85105 pos-abs">
                  <img
                    src={image2}
                    className="pos-abs  image-div bg-no-repeat fill-parent bg-cover nodeBg-85105 "
                    alt="image 331 Image"
                  />{" "}
                </div>
                <div className=" frame-46-8570 pos-abs">
                  <div className=" lorem-ipsum qui-8571 pos-abs">
                    <span className="lorem-ipsum qui-8571-0 ">{`lorem ipsum, quia dolor sit amet consectetur`}</span>
                  </div>
                </div>
              </div>
            </div>
            <div className=" frame-55-8573 pos-abs">
              <div className=" rectangle-35-8574 pos-abs"></div>
              <div className=" c-228-8575 pos-abs">
                <span className="c-228-8575-0 ">{`228`}</span>
              </div>
            </div>
            <div className=" line-7-8564 pos-abs"></div>
            <div className=" frame-59-8576 pos-abs">
              <div className=" frame-51-8577 pos-abs">
                <div className=" image-34-85106 pos-abs">
                  <img
                    src={image2}
                    className="pos-abs  image-div bg-no-repeat fill-parent bg-cover nodeBg-85106 "
                    alt="image 341 Image"
                  />{" "}
                </div>
                <div className=" frame-47-8581 pos-abs">
                  <div className=" lorem-ipsum qui-8582 pos-abs">
                    <span className="lorem-ipsum qui-8582-0 ">{`lorem ipsum, quia dolor sit amet consectetur`}</span>
                  </div>
                </div>
              </div>
            </div>
            <div className=" frame-56-8584 pos-abs">
              <div className=" rectangle-35-8585 pos-abs"></div>
              <div className=" c-150-8586 pos-abs">
                <span className="c-150-8586-0 ">{`150`}</span>
              </div>
            </div>
            <div className=" line-6-8563 pos-abs"></div>
            <div className=" frame-57-8595 pos-abs">
              <div className=" rectangle-35-8596 pos-abs"></div>
              <div className=" c-180-8597 pos-abs">
                <span className="c-180-8597-0 ">{`180`}</span>
              </div>
            </div>
            <div className=" frame-58-8587 pos-abs">
              <div className=" frame-53-8588 pos-abs">
                <div className=" image-35-85107 pos-abs">
                  <img
                    src={image2}
                    className="pos-abs  image-div bg-no-repeat fill-parent bg-cover nodeBg-85107 "
                    alt="image 351 Image"
                  />{" "}
                </div>
                <div className=" frame-45-8592 pos-abs">
                  <div className=" lorem-ipsum qui-8593 pos-abs">
                    <span className="lorem-ipsum qui-8593-0 ">{`lorem ipsum, quia dolor sit amet consectetur`}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className=" user-panel-85108 pos-abs">
            <div className=" rectangle-15-85109 pos-abs"></div>
            <div className=" frame-60-85116 pos-abs">
              <div className=" frame-52-85117 pos-abs">
                <div className=" image-33-85118 pos-abs">
                  <img
                    src={image2}
                    className="pos-abs  image-div bg-no-repeat fill-parent bg-cover nodeBg-85118 "
                    alt="image 331 Image"
                  />{" "}
                </div>
                <div className=" frame-46-85119 pos-abs">
                  <div className=" lorem-ipsum qui-85120 pos-abs">
                    <span className="lorem-ipsum qui-85120-0 ">{`lorem ipsum, quia dolor sit amet consectetur`}</span>
                  </div>
                </div>
              </div>
            </div>
            <div className=" frame-55-85122 pos-abs">
              <div className=" rectangle-35-85123 pos-abs"></div>
              <div className=" c-228-85124 pos-abs">
                <span className="c-228-85124-0 ">{`228`}</span>
              </div>
            </div>
            <div className=" line-7-85115 pos-abs"></div>
            <div className=" frame-59-85125 pos-abs">
              <div className=" frame-51-85126 pos-abs">
                <div className=" frame-47-85128 pos-abs">
                  <div className=" dimond-admin-da-85129 pos-abs">
                    <span className="dimond-admin-da-85129-0 ">{`Dimond- Admin dashboard`}</span>
                  </div>
                  <div className=" index-html-85130 pos-abs">
                    <span className="index-html-85130-0 ">{`Index. html`}</span>
                  </div>
                </div>
                <div className=" image-34-85127 pos-abs">
                  <img
                    src={image2}
                    className="pos-abs  image-div bg-no-repeat fill-parent bg-cover nodeBg-85127 "
                    alt="image 341 Image"
                  />{" "}
                </div>
              </div>
              <div className=" frame-56-85131 pos-abs">
                <div className=" rectangle-35-85132 pos-abs"></div>
                <div className=" c-150-85133 pos-abs">
                  <span className="c-150-85133-0 ">{`150`}</span>
                </div>
              </div>
            </div>
            <div className=" line-6-85114 pos-abs"></div>
            <div className=" frame-58-85134 pos-abs">
              <div className=" frame-53-85135 pos-abs">
                <div className=" frame-45-85137 pos-abs">
                  <div className=" gold-admin-dash-85138 pos-abs">
                    <span className="gold-admin-dash-85138-0 ">{`Gold- Admin dashboard`}</span>
                  </div>
                  <div className=" index-html-85139 pos-abs">
                    <span className="index-html-85139-0 ">{`Index. html`}</span>
                  </div>
                </div>
                <div className=" image-35-85136 pos-abs">
                  <img
                    src={image2}
                    className="pos-abs  image-div bg-no-repeat fill-parent bg-cover nodeBg-85136 "
                    alt="image 351 Image"
                  />{" "}
                </div>
              </div>
              <div className=" frame-57-85140 pos-abs">
                <div className=" rectangle-35-85141 pos-abs"></div>
                <div className=" c-180-85142 pos-abs">
                  <span className="c-180-85142-0 ">{`180`}</span>
                </div>
              </div>
            </div>
          </div>
        </section>
        <div className=" image-22-1918 pos-abs">
        <iframe name="marinetraffic" id="marinetraffic" width="100%" height="500" scrolling="no" frameborder="0" src="https://www.marinetraffic.com/en/ais/embed/zoom:7/centery:28.8447/centerx:-094.5703/maptype:1/shownames:false/mmsi:0/shipid:0/fleet:/fleet_id:/vtypes:/showmenu:/remember:false">Browser does not support embedded objects.<br/>Visit directly <a href="http://www.marinetraffic.com/">www.marinetraffic.com</a></iframe>4
        </div>
        {/* Frame 2981 */}
        <section className=" frame-298-16474 pos-abs">
          <div className=" menu-icon-16476 pos-abs">
            <div className=" shape-16477 pos-abs">
              <div className="nodeBg-16477 pos-abs pos-init fill-parent image-div bg-contain bg-no-repeat "></div>
            </div>
          </div>
        </section>
        <div className=" image-23-2027 pos-abs">
        <img
            src={image4}
            className="pos-abs  image-div bg-no-repeat fill-parent bg-cover nodeBg-2027 "
            alt="image 231 Image"
          />{" "}
        </div>
      </div>
    </div>
  );
};
ReactDOM.createRoot(document.getElementById("dualite-root")).render(<App />);

